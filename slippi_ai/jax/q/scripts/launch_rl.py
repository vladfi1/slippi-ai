#!/usr/bin/env python

# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing jax,
# which uses a lot of memory.
import dataclasses


if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  import os
  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
  os.environ["JAX_COMPILATION_CACHE_DIR"] = "./untracked/jax_cache"

  from absl import app, flags
  import fancyflags as ff
  import wandb

  from slippi_ai.data import chars_from_string
  from slippi_ai import flag_utils
  from slippi_ai.jax import saving, train_lib
  from slippi_ai.jax.rl import run_lib as train_rl
  from slippi_ai.jax.q import train_q_rl
  from slippi_ai.jax.agents import DType

  PP = "Platinum Player"
  DP = "Diamond Player"
  MP = "Master Player"

  NAME = MP

  CONFIG = train_q_rl.Config()

  CONFIG.runtime.max_step = 20000
  CONFIG.runtime.log_interval = 300
  CONFIG.dolphin.path = os.environ.get('MAINLINE_EXI_AI')
  CONFIG.dolphin.iso = os.environ.get('ISO_PATH')
  CONFIG.dolphin.console_timeout = 60

  CONFIG.dolphin.infinite_time = True
  CONFIG.dolphin.instant_match_restart = False  # Causes crashes :(
  CONFIG.runtime.reset_every_n_steps = 0  # Needs non-leaking dolphin build

  CONFIG.dolphin.emulation_speed = 0
  CONFIG.learner.learning_rate = 3e-5
  CONFIG.learner.q_fn_learning_rate = 1e-4
  CONFIG.learner.reward_halflife = 4
  CONFIG.learner.num_samples = 4
  # CONFIG.learner.sample_batch_size = 1

  CONFIG.learner.sample_policy_dtype = DType.FP16
  CONFIG.learner.teacher_dtype = DType.FP16
  CONFIG.learner.q_policy_dtype = DType.BF16
  CONFIG.learner.q_fn_dtype = DType.FP32
  CONFIG.agent.jax.dtype = DType.FP16
  CONFIG.opponent.other.jax.dtype = DType.FP16

  CONFIG.opponent.type = train_rl.OpponentType.SELF
  CONFIG.actor.rollout_length = 84
  CONFIG.actor.num_envs = int(os.environ.get('NUM_ENVS', 200))
  CONFIG.actor.inner_batch_size = int(os.environ.get('INNER_BATCH_SIZE', 8))
  CONFIG.actor.async_envs = True
  CONFIG.actor.num_env_steps = 4
  CONFIG.actor.gpu_inference = True
  CONFIG.agent.name = [NAME]
  CONFIG.agent.batch_steps = 4
  CONFIG.runtime.burnin_steps_after_reset = 5
  CONFIG.learner.value_burnin_steps = 100

  CONFIG_FLAG = ff.DEFINE_dict(
      'config',
      **flag_utils.get_flags_from_default(CONFIG))

  KLW = flags.DEFINE_float('klw', 0, 'Weight on KL teacher loss')

  WANDB_FLAG = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('q-rl'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
      tags=ff.StringList(['q-rl']),
  )

  DRY_RUN = flags.DEFINE_bool('dry_run', False, 'Run with fake envs and no wandb logging')

  def main(_):
    config = flag_utils.dataclass_from_dict(
        train_q_rl.Config, CONFIG_FLAG.value)

    if config.teacher is not None:
      teacher = config.teacher
    else:
      assert config.restore
      rl_state = saving.load_state_from_disk(config.restore)
      teacher = rl_state['rl_config']['teacher']
      del rl_state

    imitation_state = saving.load_state_from_disk(teacher)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config, imitation_state['config'])
    char_str = imitation_config.dataset.allowed_characters
    chars = chars_from_string(char_str)

    if config.agent.char is None:
      assert chars is not None
      config.agent.char = chars

    if config.agent.name is None:
      config.agent.name = [MP] * len(config.agent.char)

    delay = imitation_config.policy.delay
    if delay == 0:
      config.actor.num_env_steps = 0
      config.agent.batch_steps = 0

    klw = KLW.value
    config.learner.kl_teacher_weight = klw
    config.learner.reverse_kl_teacher_weight = klw

    if config.runtime.tag is None:
      if config.opponent.type is train_rl.OpponentType.SELF:
        if config.opponent.train:
          opp = 'ditto'
        elif config.opponent.update_interval is not None:
          opp = f'ditto-{config.opponent.update_interval}'
        else:
          opp = 'ditto-fixed'
      elif config.opponent.type is train_rl.OpponentType.CPU:
        opp = 'vs_cpu'
      elif config.opponent.type is train_rl.OpponentType.OTHER:
        # assert config.opponent.other.path is not None
        # opponent_state = saving.load_state_from_disk(config.opponent.other.path)
        opp = 'vs-fixed'
      else:
        raise ValueError(f"Unsupported opponent type: {config.opponent.type}")

      fs = imitation_config.policy.frame_skip
      ns = config.learner.num_samples
      if config.learner.include_action_taken_in_samples:
        ns += 1
      klw_str = f"_klw{klw:.0e}" if klw > 0 else ""
      ep = config.learner.epoch_length

      lr = config.learner.learning_rate

      wba = f"_wba" if config.learner.weight_by_advantage else ""
      config.runtime.tag = f"qrl_{char_str}_d{delay}_{opp}{klw_str}_rfs{fs}_ns{ns}_ep{ep}_lr{lr:.0e}{wba}"

    wandb_kwargs = dict(WANDB_FLAG.value)
    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.runtime.tag

    if DRY_RUN.value:
      wandb_kwargs['mode'] = 'disabled'
      config.actor.use_fake_envs = True
      config.actor.use_sim_envs = False  # fake sim envs give bad data
      config.runtime.log_interval = 20
      config.runtime.save_interval = -1

    wandb.init(config=dataclasses.asdict(config), **wandb_kwargs)

    train_q_rl.run(config)

  app.run(main)
