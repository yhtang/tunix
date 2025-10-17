# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO learner."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence

import contextlib
import flax
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_helpers

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  pass


@dataclasses.dataclass(slots=True, kw_only=True)
class GRPOConfig:
  """Configuration for GRPO algorithm.

  Parameters:
    num_generations: The number of times the policy generates multiple
      responses for a given prompt within a single training step. This
      corresponds to 'G' in Algorithm 1 in the paper. A higher value means
      more samples are used to compute relative advantages.
    num_iterations: The number of iterations per batch (𝜇 in GRPO algo 1).
    beta: The coefficient for the KL divergence penalty (𝛽) in the GRPO loss
      function. This term prevents policy updates from deviating too far from
      the reference model. A value of 0.0 means no KL penalty is applied.
    epsilon: Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to
      PPO, it ensures stable updates.
    loss_algo: use GRPO or GSPO for loss computation. GRPO loss is per-batch
      normalized instead of per-response normalized as mentioned in the
      paper. For GSPO, we use gspo-token loss which is more flexible.

  References:
    - GRPO: https://arxiv.org/abs/2402.03300
    - GSPO: https://www.arxiv.org/pdf/2507.18071
  """

  num_generations: int = 2
  num_iterations: int = 1
  beta: float = 0.04
  epsilon: float = 0.2
  loss_algo: str = "grpo"  # grpo or gspo-token

  def __post_init__(self):
    if self.num_generations <= 1:
      raise ValueError(
          "num_generations must be greater than 1. Received: "
          f"{self.num_generations}"
      )
    if self.loss_algo not in ["grpo", "gspo-token"]:
      raise ValueError(
          "loss_algo should be either grpo or gspo-token. Received: "
          f"{self.loss_algo}"
      )


class GRPOLearner(rl_learner.RLLearner):
  """GRPO (Group Relative Policy Optimization) learner.

  GRPO is a reinforcement learning algorithm designed to enhance the reasoning
  abilities of large language models, like mathematical problem-solving. It is
  a variant of Proximal Policy Optimization (PPO) that reduces memory usage by
  eliminating the need for a separate value function model. GRPO works by
  generating multiple responses for a given prompt, evaluating these responses
  using a reward model, and then calculating a relative advantage based on the
  group's performance to update the policy.

  References:
    - https://arxiv.org/abs/2402.03300
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      reward_fns: RewardFn | List[RewardFn],
      grpo_config: GRPOConfig,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `GRPOTrainer`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept `prompts`, `completions` and optional keyword arguments, and
        return a list of float rewards.
      grpo_config: An instance of `GRPOConfig` containing all GRPO specific
        parameters.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept ``prompts``, ``completions``,
        ``rewards``, ``advantages`` and optional keyword arguments, and return
        a dictionary of metric names to tuples of
        ``(metric_value, aggregation_fn)``:

           >>> def metric_fn(
           ...     prompts, completions, rewards, advantages, **kargs
           ... ):
           ...     return {
           ...       # ...
           ...       "prompt_min_len": (min(len(p) for p in prompts), np.min),
           ...       # ... }
      data_shuffle_seed: The seed used to shuffle the training data.
    """  # fmt: skip
    self.grpo_config = grpo_config
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    # Workaround for passing in importance_sampling_algo as jax transforms
    # doesn't like partial functions with kwargs.
    loss_fn = lambda model, train_example, beta, epsilon: grpo_loss_fn(
        model,
        train_example,
        beta=beta,
        epsilon=epsilon,
        loss_algo=self.grpo_config.loss_algo,
    )

    self.rl_cluster.actor_trainer.with_loss_fn(
        loss_fn,
        has_aux=True,
    )
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "beta": self.grpo_config.beta,
            "epsilon": self.grpo_config.epsilon,
        }
    )
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
    self.rl_cluster.actor_trainer.with_tqdm_metrics_to_display([
        lambda: "kl" if self.grpo_config.beta != 0.0 else None,
    ])

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generate text completions and compute the advantages for GRPO training.

    Args:
      training_input: A dictionary containing the training input data,
        containing the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data, including
      prompt IDs, completion IDs, masks, advantages, and per-token log
      probabilities from the reference and policy models.
    """
    training_input["prompts"] = list(training_input["prompts"])
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            self._rollout_micro_batch_size * self.grpo_config.num_generations
        ),
    )
    completion_ids = rollout_output.tokens
    prompt_ids = rollout_output.left_padded_prompt_tokens
    completion_text = rollout_output.text

    # Assemble masks
    prompt_mask = prompt_ids != pad_value
    completion_padding_mask = jnp.not_equal(completion_ids, pad_value)
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    if self.grpo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
          micro_batch_size=(
              self._compute_logps_micro_batch_size
              * self.grpo_config.num_generations
          ),
      )
    else:
      ref_per_token_logps = None
    if self.grpo_config.num_iterations > 1:
      old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          micro_batch_size=(
              self._compute_logps_micro_batch_size
              * self.grpo_config.num_generations
          ),
      )
    else:
      old_per_token_logps = None

    # Compute rewards and advantages
    rewards = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_text,
        mode=mode,
        **{k: v for k, v in training_input.items() if k != "prompts"},
    )

    advantages = grpo_helpers.compute_advantages(
        rewards, self.grpo_config.num_generations
    )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=completion_text,
          advances=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is a string of format {row_offset}_{group_offset} where
    row_offset is the row index of the example data source and
    group_offset is the group index of the example in the generation group.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self.grpo_config.num_generations
    row_offset = steps * batch_size
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.grpo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.grpo_config.num_generations),
        batch_size,
    )
    return [
        f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)
    ]

  def _num_iterations(self) -> int:
    return self.grpo_config.num_iterations

  def _num_generations(self) -> int:
    return self.grpo_config.num_generations

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """GRPO training loop.

    Algorithm as below: extract from https://arxiv.org/abs/2402.03300 ::

        Input:
            initial policy model πθinit;
            reward models rφ;
            task prompts D;
            hyperparameters ε, β, μ

        policy model πθ ← πθinit
        for iteration = 1, ..., I do
          reference model πref ← πθ
          for step = 1, ..., M do
            Sample a batch D♭ from D
            Update the old policy model πθold ← πθ
            Sample G outputs {oi}G_i=1 ~ πθold(· | q) for each question q ∈ D♭
            Compute rewards {ri}G_i=1 for each sampled output oi by running rφ
            Compute Âi,t for the t-th token of oi through group relative
            advantage estimation.
            for GRPO iteration = 1, ..., μ do
              Update the policy model πθ by maximizing the GRPO objective
              (Equation 21)
          Update rφ through continuous training using a replay mechanism.
        Output πθ

    .. note::

        1. The outer loop (I) is ignored for now because we never update the
           reference model for now.

        2. Currently sample and train hold the same referece to the model. So
           we also omit the step to update the sampler model.

    Args:
      train_ds: An iterable of training input data, where each element is a
        dictionary containing the key 'prompts'.
      eval_ds: An iterable of evaluation input data, where each element is a
        dictionary containing the key 'prompts'.
      skip_jit: Whether to skip JIT compilation of the training loop.
    """
    super().train(train_ds, eval_ds, skip_jit)


def grpo_loss_fn(model, train_example, beta, epsilon, loss_algo):
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    beta: The coefficient for the KL divergence penalty. A value of 0.0 means
      no KL penalty is applied.
    epsilon: Epsilon value for clipping.
    loss_algo: The loss algorithm to use. Can be grpo or gspo-token.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  prompt_ids, prompt_mask = (
      train_example.prompt_ids,
      train_example.prompt_mask,
  )
  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  input_ids = jnp.concat([prompt_ids, completion_ids], axis=1)
  prompt_completion_mask = jnp.concat([prompt_mask, completion_mask], axis=-1)
  attention_mask = common.make_causal_attn_mask(prompt_completion_mask)
  logits_to_keep = completion_ids.shape[1]
  positions = common.build_positions_from_mask(prompt_completion_mask)

  per_token_logps, _ = common.get_per_token_logps(
      model,
      input_tokens=input_ids,
      positions=positions,
      attn_mask=attention_mask,
      logits_to_keep=logits_to_keep,
  )
  advantages = train_example.advantages

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = train_example.old_per_token_logps

  seq_importance_ratio = per_token_logps - old_per_token_logps
  if loss_algo == "gspo-token":
    seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
        axis=-1
    ) / jnp.clip(completion_mask.sum(-1), min=1)
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_importance_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)

  coef_1 = jnp.exp(seq_importance_ratio)
  coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon)

  # TODO(tsbao): We should handle token level advantages.
  per_token_loss = -jnp.minimum(
      coef_1 * jnp.expand_dims(advantages, 1),
      coef_2 * jnp.expand_dims(advantages, 1),
  )

  if loss_algo == "gspo-token":
    loss_denominator = jnp.clip(completion_mask.sum(axis=-1), min=1)
  else:  # grpo
    loss_denominator = jnp.clip(completion_mask.sum(), min=1)

  aux = {"kl": 0.0}
  if beta != 0.0:
    kl = common.compute_kl_divergence(
        per_token_logps, train_example.ref_per_token_logps
    )
    per_token_loss = per_token_loss + beta * kl

    # Log mean KL.
    aux["kl"] = (kl * completion_mask).sum() / loss_denominator.mean()

  if loss_algo == "gspo-token":
    loss = (
        (per_token_loss * completion_mask).sum(axis=-1) / loss_denominator
    ).mean()
  else:  # grpo
    loss = (per_token_loss * completion_mask).sum() / loss_denominator

  return loss, aux

GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
