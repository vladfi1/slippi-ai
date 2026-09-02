#!/usr/bin/env python

# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.

if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  import logging
  import os

  os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1'

  from absl import app, flags
  import fancyflags as ff
  import wandb

  from slippi_ai import flag_utils
  from slippi_ai.jax import saving, train_lib
  from slippi_ai.jax.rl import train_two_lib
  from slippi_ai.jax.agents import DType

  PGW=3

  CONFIG = train_two_lib.Config()

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
  CONFIG.learner.ppo.num_epochs=2
  CONFIG.learner.ppo.num_batches=1
  CONFIG.learner.ppo.beta=3e-1
  CONFIG.learner.ppo.epsilon=1e-2
  CONFIG.p1.batch_steps=4
  CONFIG.p2.batch_steps=4
  CONFIG.actor.rollout_length=80
  CONFIG.actor.use_sim_envs=True
  CONFIG.actor.num_envs=2048
  CONFIG.learner.microbatch_size=512
  CONFIG.learner.value_mbs=1024
  CONFIG.actor.inner_batch_size=-1
  CONFIG.actor.async_envs=True
  CONFIG.actor.num_env_steps=4
  CONFIG.actor.gpu_inference=True
  CONFIG.runtime.burnin_steps_after_reset=5
  CONFIG.runtime.reset_every_n_steps=None
  CONFIG.learner.optimizer_burnin_epochs=0
  CONFIG.learner.value_burnin_epochs=0

  # Optimal dtypes
  CONFIG.p1.jax.dtype = DType.FP16
  CONFIG.p2.jax.dtype = DType.FP16
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

  PERF = flags.DEFINE_bool('perf', False, 'Run to measure performance')

  def get_imitation_config(config_path: str) -> train_lib.Config:
    imitation_state = saving.load_state_from_disk(config_path)
    imitation_config = flag_utils.dataclass_from_dict(
        train_lib.Config, saving.upgrade_config(imitation_state['config']))
    return imitation_config

  def main(_):
    learner_kwargs = CONFIG_FLAG.value['learner']
    learner_kwargs.update(
        kl_teacher_weight=KLW.value,
        reverse_kl_teacher_weight=KLW.value,
    )

    CONFIG_FLAG.value['learner1'] = flag_utils.override_dict(
        learner_kwargs, CONFIG_FLAG, ['learner1'])
    CONFIG_FLAG.value['learner2'] = flag_utils.override_dict(
        learner_kwargs, CONFIG_FLAG, ['learner2'])

    config = flag_utils.dataclass_from_dict(
        train_two_lib.Config, CONFIG_FLAG.value)

    p1_imitation_config = get_imitation_config(config.p1.teacher)
    p2_imitation_config = get_imitation_config(config.p2.teacher)

    c1 = p1_imitation_config.dataset.allowed_characters
    c2 = p2_imitation_config.dataset.allowed_characters

    d1 = p1_imitation_config.policy.delay
    d2 = p2_imitation_config.policy.delay
    if d1 != d2:
      logging.warning('Teachers must have the same delay.')

    if d1 == d2:
      config.runtime.tag = f"{c1}_vs_{c2}_d{d1}_kl_{KLW.value:.0e}"
    else:
      config.runtime.tag = f"{c1}_d{d1}_vs_{c2}_d{d2}_kl_{KLW.value:.0e}"

    wandb_kwargs = dict(WANDB_FLAG.value)

    if PERF.value:
      config.runtime.save_interval = -1
      config.runtime.log_interval = 20
      config.learner.optimizer_burnin_epochs = 0
      config.learner.value_burnin_epochs = 0
      # config.actor.use_sim_envs = False
      # config.actor.use_fake_envs = True
      wandb_kwargs['mode'] = 'disabled'

    if wandb_kwargs['name'] is None:
      wandb_kwargs['name'] = config.runtime.tag

    if config.actor.use_fake_envs:
      wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        config=CONFIG_FLAG.value,
        **wandb_kwargs,
    )

    train_two_lib.run(config)

  app.run(main)
