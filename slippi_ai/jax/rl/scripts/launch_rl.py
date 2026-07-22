#!/usr/bin/env python

# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.
if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  import logging
  import os

  from absl import app, flags
  import fancyflags as ff
  import wandb

  import melee

  from slippi_ai.data import chars_from_string
  from slippi_ai import flag_utils
  from slippi_ai.jax import saving, train_lib
  from slippi_ai.jax.rl import run_lib
  from slippi_ai.jax.agents import DType

  PP="Platinum Player"
  DP="Diamond Player"
  MP="Master Player"

  NAME = MP

  PGW=3

  CONFIG = run_lib.Config()

  CONFIG.runtime.max_step=10000
  CONFIG.runtime.log_interval=300
  CONFIG.dolphin.path=os.environ.get('MAINLINE_EXI_AI')
  CONFIG.dolphin.iso=os.environ.get('ISO_PATH')
  CONFIG.dolphin.console_timeout=60
  CONFIG.dolphin.infinite_time=False  # regularly randomize stages
  CONFIG.dolphin.emulation_speed=0
  CONFIG.learner.learning_rate=3e-5
  CONFIG.learner.value_cost=1
  CONFIG.learner.policy_gradient_weight=PGW
  CONFIG.learner.reward_halflife=4
  CONFIG.learner.ppo.num_epochs=2
  CONFIG.learner.ppo.num_batches=16
  CONFIG.learner.ppo.beta=3e-1
  CONFIG.learner.ppo.epsilon=1e-2
  # CONFIG.teacher=f'pickled_models/jax/{MODEL}'
  CONFIG.opponent.type=run_lib.OpponentType.SELF
  CONFIG.opponent.train=True
  CONFIG.actor.rollout_length=80
  CONFIG.actor.num_envs=200
  CONFIG.actor.inner_batch_size=8
  CONFIG.actor.async_envs=True
  CONFIG.actor.num_env_steps=4
  CONFIG.actor.gpu_inference=True
  CONFIG.agent.name=[NAME]
  CONFIG.agent.batch_steps=4
  CONFIG.runtime.burnin_steps_after_reset=5
  CONFIG.runtime.reset_every_n_steps=512
  CONFIG.learner.optimizer_burnin_epochs=8
  CONFIG.learner.value_burnin_epochs=8

  # Optimal dtypes
  CONFIG.agent.jax.dtype = DType.FP16
  CONFIG.learner.teacher_dtype = DType.FP16
  CONFIG.learner.value_dtype = DType.BF16
  CONFIG.learner.policy_dtype = DType.FP16

  CONFIG_FLAG = ff.DEFINE_dict(
      'config',
      **flag_utils.get_flags_from_default(CONFIG))

  WANDB_FLAG = ff.DEFINE_dict(
      'wandb',
      project=ff.String('slippi-ai'),
      mode=ff.Enum('online', ['online', 'offline', 'disabled']),
      group=ff.String('rl'),
      name=ff.String(None),
      notes=ff.String(None),
      dir=ff.String(None, 'directory to save logs'),
      tags=ff.StringList(['ppo']),
  )

  KLW = flags.DEFINE_float('kl_weight', 1e-2, 'weight for KL teacher losses')

  LEARNER_PERF = flags.DEFINE_bool('learner_perf', False, 'Run to measure learner performance')

  def main(_):
    config = flag_utils.dataclass_from_dict(
        run_lib.Config, CONFIG_FLAG.value)

    config.learner.kl_teacher_weight = KLW.value
    config.learner.reverse_kl_teacher_weight = KLW.value


    if config.teacher is not None:
      imitation_state = saving.load_state_from_disk(config.teacher)
      imitation_config = flag_utils.dataclass_from_dict(
          train_lib.Config, imitation_state['config'])
      char_str = imitation_config.dataset.allowed_characters
      chars = chars_from_string(char_str)

      if config.agent.char is None:
        assert chars is not None
        config.agent.char = chars
        logging.info(f"Using teacher's allowed characters: {chars}")
      elif chars is not None:
        for char in config.agent.char:
          if char not in chars:
            raise ValueError(f"Character {char} not in teacher's allowed characters: {chars}")

      if config.agent.name is None:
        config.agent.name = [MP] * len(config.agent.char)

      delay = imitation_config.policy.delay

      if config.runtime.tag is None:
        if config.opponent.type is run_lib.OpponentType.SELF:
          if config.opponent.train:
            opp = 'ditto'
          elif config.opponent.update_interval is not None:
            opp = f'ditto-{config.opponent.update_interval}'
          else:
            opp = 'ditto-fixed'
        elif config.opponent.type is run_lib.OpponentType.CPU:
          opp = 'vs_cpu'
        elif config.opponent.type is run_lib.OpponentType.OTHER:
          # assert config.opponent.other.path is not None
          # opponent_state = saving.load_state_from_disk(config.opponent.other.path)
          opp = 'vs-fixed'
        else:
          raise ValueError(f"Unsupported opponent type: {config.opponent.type}")

        config.runtime.tag = f"{char_str}_d{delay}_{opp}_kl_{KLW.value:.0e}"

    wandb_kwargs = dict(WANDB_FLAG.value)

    if LEARNER_PERF.value:
      config.runtime.save_interval = -1
      config.runtime.log_interval = 20
      config.learner.optimizer_burnin_epochs = 0
      config.learner.value_burnin_epochs = 0
      # config.actor.use_sim_envs = False
      # config.actor.use_fake_envs = True
      wandb_kwargs['mode'] = 'disabled'
      config.agent

    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.runtime.tag

    wandb.init(
        config=CONFIG_FLAG.value,
        **wandb_kwargs,
    )

    run_lib.run(config)

  app.run(main)
