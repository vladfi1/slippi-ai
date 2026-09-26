#!/usr/bin/env python
"""Train ten characters against each other with train_many.

Everything is configured in python: edit AGENTS below to change per-agent
names, ratings and learner settings, and BASE_* for the shared settings.
Only a few runtime switches are exposed as flags.
"""

# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing JAX, which uses a lot of memory.
if __name__ == '__main__':
  __spec__ = None  # https://github.com/python/cpython/issues/87115

  import copy
  import dataclasses
  import os
  import pathlib

  # Must be set before jax is imported (via slippi_ai below): jax reads
  # JAX_COMPILATION_CACHE_DIR into its config at import time.
  os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "./untracked/jax_cache")
  os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.95')
  # The default BFC allocator fragments under this workload's mix of
  # multi-GiB trajectory buffers and per-step actor outputs: with 4 agents x
  # 512 envs it fails to find a contiguous 1.8 GiB block for the teacher
  # unroll with 5 GiB free. The CUDA async allocator (a VMM-backed pool)
  # doesn't have this problem.
  os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'cuda_async')

  from absl import app, flags
  import fancyflags as ff
  import wandb

  from slippi_ai import nametags, flag_utils
  from slippi_ai.jax.agents import DType
  from slippi_ai.jax.rl import train_many_lib

  MODELS_DIR = pathlib.Path('pickled_models/jax')
  MP = nametags.DEFAULT_NAME  # 'Master Player'

  # ---------------------------------------------------------------------------
  # Shared settings, mostly following scripts/launch_two.py.
  # ---------------------------------------------------------------------------

  CONFIG = train_many_lib.Config()

  CONFIG.runtime.max_step=50000
  CONFIG.runtime.log_interval=300
  CONFIG.dolphin.path=os.environ.get('MAINLINE_EXI_AI')
  CONFIG.dolphin.iso=os.environ.get('ISO_PATH')
  CONFIG.dolphin.console_timeout=60
  CONFIG.dolphin.infinite_time=False  # regularly randomize stages
  CONFIG.dolphin.emulation_speed=0
  CONFIG.learner.learning_rate=2e-5
  CONFIG.learner.value_cost=1
  CONFIG.learner.policy_gradient_weight=3
  CONFIG.learner.ppo.num_epochs=2
  CONFIG.learner.ppo.num_batches=1
  CONFIG.learner.ppo.beta=3e-1
  CONFIG.learner.ppo.epsilon=1e-2
  CONFIG.actor.rollout_length=80
  CONFIG.actor.use_sim_envs=True
  CONFIG.actor.num_envs=256
  CONFIG.learner.microbatch_size=1024
  CONFIG.learner.value_mbs=1024
  # Unmicrobatched (0), the teacher unroll over a 4096-trajectory batch needs
  # a 2.3 GiB temp buffer; this costs nothing since it takes no gradients.
  CONFIG.learner.teacher_mbs=1024
  CONFIG.actor.inner_batch_size=-1
  CONFIG.actor.async_envs=True
  CONFIG.actor.num_env_steps=4
  CONFIG.actor.gpu_inference=True
  CONFIG.runtime.burnin_steps_after_reset=5
  CONFIG.runtime.reset_every_n_steps=None
  CONFIG.learner.optimizer_burnin_epochs=0
  CONFIG.learner.value_burnin_epochs=0

  # Optimal dtypes
  CONFIG.learner.teacher_dtype = DType.FP16
  CONFIG.learner.value_dtype = DType.BF16
  CONFIG.learner.policy_dtype = DType.FP16
  CONFIG.agent.jax.dtype = DType.FP16

  CONFIG.agent.rating = 3200
  CONFIG.agent.batch_steps = 4

  # ---------------------------------------------------------------------------
  # Per-agent settings. Each entry lists the teacher checkpoint and any fields
  # of BASE_AGENT / BASE_LEARNER to override for that agent. Nested learner
  # overrides (e.g. ppo) take a dict.
  #
  # The "_noname" teachers were trained without nametags, so they ignore
  # `name`; the others accept any name in their name_map (see name_map.json
  # next to the checkpoint, or the "Master Player" etc. skill tiers).
  # ---------------------------------------------------------------------------

  @dataclasses.dataclass
  class Entry:
    teacher: str
    agent: dict = dataclasses.field(default_factory=dict)
    kl_weight: float = 1

  AGENTS = [
      Entry(
          teacher='fox_d21_tx3x1024_rating',
          agent=dict(name=['Cody', 'Hax', 'Aklo', 'SFAT', 'Zamu']),
          kl_weight=2,
      ),
      Entry(
          teacher='falco_d21_tx3x1024_rating',
          agent=dict(name=['Ginger', 'BBB', 'Frenzy', 'KJH']),
      ),
      Entry(
          teacher='marth_d21_tx3x1024_rating',
          agent=dict(name=['Zain', 'Kodorin']),
      ),
      Entry(
          teacher='sheik_d21_tx3x1024_rating',
          agent=dict(name=['Krudo', 'Jmook']),
      ),
      Entry(
          teacher='jigglypuff_d21_tx3x1024_rating_noname',
      ),
      Entry(
          teacher='peach_d21_tx3x1024_rating_noname',
      ),
      Entry(
          teacher='falcon_d21_tx3x1024_rating_noname',
      ),
      Entry(
          teacher='ics_d21_tx3x1024_rating_noname',
          kl_weight=0.5,
      ),
      Entry(
          teacher='yoshi_d21_tx3x1024_rating',
          agent=dict(name=['Amsa']),
          kl_weight=0.5,
      ),
      Entry(
          teacher='luigi_d21_tx3x1024_rating',
          agent=dict(name=['JahRidin', 'RapM']),
          kl_weight=0.5,
      ),
  ]

  def make_agent_spec(config: train_many_lib.Config, entry: Entry) -> train_many_lib.AgentSpec:
    agent = copy.deepcopy(config.agent)
    agent.teacher = str(MODELS_DIR / entry.teacher)
    agent = dataclasses.replace(agent, **entry.agent)

    learner = copy.deepcopy(config.learner)
    learner.kl_teacher_weight *= entry.kl_weight
    learner.reverse_kl_teacher_weight *= entry.kl_weight
    return train_many_lib.AgentSpec(
        agent=agent,
        learner=learner,
    )

  # ---------------------------------------------------------------------------
  # Runtime flags.
  # ---------------------------------------------------------------------------

  CONFIG_FLAG = ff.DEFINE_dict(
      'config',
      **flag_utils.get_flags_from_default(CONFIG))

  KLW = flags.DEFINE_float('kl_weight', 1e-2, 'weight for KL teacher losses')
  AGENT_LIMIT = flags.DEFINE_integer('agent_limit', None, 'Limit number of agents to train.')

  FAKE_ENVS = flags.DEFINE_bool('fake_envs', False, 'Use fake envs (testing).')
  PERF = flags.DEFINE_bool('perf', False, 'Run to measure performance.')

  WANDB_MODE = flags.DEFINE_enum(
      'wandb_mode', 'online', ['online', 'offline', 'disabled'], 'wandb mode.')
  WANDB_NAME = flags.DEFINE_string('wandb_name', None, 'wandb run name.')
  WANDB_NOTES = flags.DEFINE_string('wandb_notes', None, 'wandb notes.')

  def main(_):
    config = flag_utils.dataclass_from_dict(
        train_many_lib.Config, CONFIG_FLAG.value)
    config.learner.kl_teacher_weight = KLW.value
    config.learner.reverse_kl_teacher_weight = KLW.value

    if FAKE_ENVS.value:
      # Tiny synchronous setup for testing the plumbing.
      config.actor.async_envs = False
      config.actor.num_env_steps = 0
      config.actor.inner_batch_size = 1

    agent_specs = [make_agent_spec(config, entry) for entry in AGENTS]
    if AGENT_LIMIT.value is not None:
      agent_specs = agent_specs[:AGENT_LIMIT.value]

    if FAKE_ENVS.value:
      # Microbatching breaks when the microbatch exceeds the (tiny) batch, and
      # the actor KL check is meaningless on fake data.
      for spec in agent_specs:
        spec.learner.microbatch_size = 0
        spec.learner.value_mbs = 0
        spec.learner.ppo.max_mean_actor_kl = float('inf')

    if config.runtime.tag is None:
      config.runtime.tag = f'many{len(agent_specs)}_klw{KLW.value:.0e}'

    wandb_kwargs = dict(
        project='slippi-ai',
        mode=WANDB_MODE.value,
        group='jax-rl-many',
        name=WANDB_NAME.value or config.runtime.tag,
        notes=WANDB_NOTES.value,
        tags=['ppo', 'many'],
    )

    if PERF.value:
      config.runtime.save_interval = -1
      config.runtime.log_interval = 20
      wandb_kwargs['mode'] = 'disabled'

    if FAKE_ENVS.value:
      wandb_kwargs['mode'] = 'disabled'

    wandb.init(
        config=dict(
            config=dataclasses.asdict(config),
            agents=[dataclasses.asdict(spec) for spec in agent_specs],
        ),
        **wandb_kwargs,
    )

    train_many_lib.run(config, agent_specs)

  app.run(main)
