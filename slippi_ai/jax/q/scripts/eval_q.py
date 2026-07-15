#!/usr/bin/env python
"""Evaluate a Q-function's argmax distribution over policy samples.

Analogous to nash/scripts/eval_nash_q.py: runs a single evaluation (no
training) of train_q_policy with a frozen sample policy and q-function. The
main quantity of interest is the entropy of the distribution over sampled
actions induced by the per-epistemic-index argmaxes, which measures how
(un)certain the q-function is about the best action. Compare across
q-functions by varying --config.initialize_q_function_from.
"""

import dataclasses
import math
import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "./untracked/jax_cache"

from absl import app, flags
import wandb
import fancyflags as ff

from slippi_ai import flag_utils, paths
from slippi_ai.jax import saving, train_lib
from slippi_ai.jax.q import train_q_policy
from slippi_ai.jax.q import q_policy_learner as learner_lib
from slippi_ai.jax.agents import DType

def default_config():
  config = train_q_policy.Config()

  config.runtime.max_eval_steps = 50
  config.runtime.run_single_eval = True
  config.runtime.verbose_eval = True

  config.data.batch_size = 256
  config.data.unroll_length = 84
  config.data.num_workers = 1

  config.learner.num_samples = 3
  config.learner.num_index_samples = 16
  config.learner.sample_policy_dtype = DType.FP16
  config.learner.q_policy_dtype = DType.FP16

  config.dataset.mirror = False
  config.dataset.data_dir = os.environ.get("DATA_DIR")
  config.dataset.meta_path = os.environ.get("META_PATH")

  return config

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  TOY_DATA = flags.DEFINE_bool('toy_data', False, 'Use toy data for quick testing')

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(default_config()))

  # passed to wandb.init
  WANDB = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('q-eval'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
  )

  def main(_):
    config = flag_utils.dataclass_from_dict(train_q_policy.Config, CONFIG.value)

    assert config.initialize_policies_from is not None
    imitation_state = saving.load_state_from_disk(config.initialize_policies_from)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config,
        saving.upgrade_config(imitation_state['config']))
    del imitation_state

    assert config.initialize_q_function_from is not None

    if TOY_DATA.value:
      config.dataset.data_dir = str(paths.TOY_DATA_DIR)
      config.dataset.meta_path = str(paths.TOY_META_PATH)
      config.dataset.test_ratio = 0.5
      char = 'all'
      config.data.cached = True
      config.data.num_workers = 0
      config.runtime.max_eval_steps = 1
      if config.tag is None:
        config.tag = 'toy_eval'
    else:
      char = imitation_config.dataset.allowed_characters

      if config.tag is None:
        q_fn_name = os.path.basename(config.initialize_q_function_from)
        if q_fn_name == 'latest.pkl':
          q_fn_name = os.path.basename(os.path.dirname(config.initialize_q_function_from))
        config.tag = (
            f'{q_fn_name}_ns{config.learner.num_samples}'
            f'_ni{config.learner.num_index_samples}_eval')

    config.dataset.allowed_characters = char

    wandb_kwargs = dict(WANDB.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.tag
      if TOY_DATA.value:
        wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        config=dataclasses.asdict(config),
        **wandb_kwargs,
    )
    mean_stats = train_q_policy.train(config)
    assert mean_stats is not None

    q_fn_stats = mean_stats[learner_lib.Q_FUNCTION]

    num_actions = config.learner.num_samples
    if config.learner.include_action_taken_in_samples:
      num_actions += 1

    print(f'=== {config.tag} ===')
    # Entropy of the per-index argmax distribution; 0 means all epistemic
    # indices agree on the best action. Also capped by the number of sampled
    # indices at log(num_index_samples).
    print(f'argmax entropy: {q_fn_stats["entropy"]:.4f}'
          f' (max {math.log(num_actions):.4f} with {num_actions} actions)')

    for key in [
        'action_taken_is_optimal',
        'optimal_advantages',
        'sample_policy_advantages',
        'q_bias',
    ]:
      print(f'{key}: {q_fn_stats[key]:.4f}')

    print(f'q uev: {q_fn_stats["q"]["uev"]:.4f}'
          f' (ensemble: {q_fn_stats["ensemble"]["q"]["uev"]:.4f})')

    # Cross-entropy from the argmax distribution to the (imitation-initialized)
    # q_policy; directly comparable to the argmax entropy above.
    print(f'q_policy argmax cross-entropy:'
          f' {mean_stats[learner_lib.Q_POLICY]["q_loss"]:.4f}')

  app.run(main)
