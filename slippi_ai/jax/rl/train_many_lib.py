"""Train many JAX agents against each other.

Generalizes train_two to an arbitrary number of agents, each initialized from
its own teacher. Every pair of agents gets its own rollout worker, including
(optionally) each agent paired with itself. A cross-play worker only yields
one trajectory per env for each of its two agents, while a self-play worker
yields two for its single agent, so cross-play workers run twice as many envs.
Each agent therefore sees 2 * num_envs trajectories per worker it takes part
in, for a learner batch size of 2 * num_envs * num_agents (with self-play).

Training alternates between stepping all of the rollout workers and stepping
all of the learners, each learner training on the concatenation of its
trajectories from every worker it participates in.
"""

import contextlib
import dataclasses
import itertools
import logging
import os
import resource
import typing as tp

import melee
import numpy as np
import jax
import jax.numpy as jnp

from slippi_ai import (
    dolphin as dolphin_lib,
    evaluators,
    nametags,
    reward,
    utils,
)
from slippi_ai.jax import train_lib as jax_train_lib
from slippi_ai.jax.rl import learner as learner_lib
from slippi_ai.jax.rl import run_lib
from slippi_ai.jax.rl import train_two_lib
from slippi_ai.types import Game

field = lambda f: dataclasses.field(default_factory=f)


@dataclasses.dataclass
class RuntimeConfig(train_two_lib.RuntimeConfig):
  expt_root: str = 'experiments/jax/train_many'

@dataclasses.dataclass
class Config:
  runtime: RuntimeConfig = field(RuntimeConfig)

  dolphin: dolphin_lib.DolphinConfig = field(dolphin_lib.DolphinConfig)
  learner: learner_lib.LearnerConfig = field(
      train_two_lib.default_learner_config)
  actor: run_lib.ActorConfig = field(run_lib.ActorConfig)
  agent: train_two_lib.AgentConfig = field(train_two_lib.AgentConfig)

  # One agent per teacher. May be omitted when restoring from expt_dir.
  teachers: list[str] = field(list)

  # The per-agent lists below must be empty (unset), have a single entry
  # (shared by all agents), or have one entry per teacher.

  # Will override a value function in the teacher checkpoint. Necessary if the
  # teacher has no value function i.e. was produced by policy-only training.
  value_functions: list[str] = field(list)
  # Character to play. If unset, inferred from the teacher checkpoint (only
  # works if it was trained on a single character).
  chars: list[melee.Character] = field(list)
  # Rating to condition the agent on. Required when the teacher was trained
  # with ratings (embed.with_rating); ignored otherwise.
  ratings: list[float] = field(list)


@dataclasses.dataclass
class AgentSpec:
  """Full per-agent settings. Can't be expressed as flags; for use from python
  (see scripts/train_many_10.py), where it takes the place of Config.teachers,
  Config.agent, Config.learner and the per-agent lists."""
  agent: train_two_lib.AgentConfig
  learner: learner_lib.LearnerConfig


DEFAULT_CONFIG = Config()
DEFAULT_CONFIG.dolphin.console_timeout = 30

Logger = run_lib.Logger

PORTS = (1, 2)
NUM_AGENTS_FILE = 'num_agents.txt'
MINUTES_PER_FRAME = 60 * 60

T = tp.TypeVar('T')


def _per_agent(values: list[T], num_agents: int, name: str) -> list[tp.Optional[T]]:
  if not values:
    return [None] * num_agents
  if len(values) == 1:
    return list(values) * num_agents
  if len(values) != num_agents:
    raise ValueError(
        f'{name} must have 0, 1 or {num_agents} entries, got {len(values)}.')
  return list(values)


def get_num_agents(num_teachers: int, expt_dir: str) -> int:
  """The number of agents, which is recorded in expt_dir for restoring."""
  path = os.path.join(expt_dir, NUM_AGENTS_FILE)
  previous = None
  if os.path.exists(path):
    with open(path) as f:
      previous = int(f.read())

  num_agents = num_teachers or previous
  if not num_agents:
    raise ValueError('Must pass "teachers" if not restoring.')
  if previous is not None and previous != num_agents:
    raise ValueError(
        f'Got {num_agents} teachers but {expt_dir} has {previous} agents.')

  with open(path, 'w') as f:
    f.write(str(num_agents))
  return num_agents


def get_agent_specs(config: Config, num_agents: int) -> list[AgentSpec]:
  """Expands the (flag-friendly) config into full per-agent settings."""
  teachers = _per_agent(config.teachers, num_agents, 'teachers')
  value_functions = _per_agent(
      config.value_functions, num_agents, 'value_functions')
  chars = _per_agent(config.chars, num_agents, 'chars')
  ratings = _per_agent(config.ratings, num_agents, 'ratings')

  return [
      AgentSpec(
          agent=train_two_lib.AgentConfig(
              teacher=teachers[i],
              value_function=value_functions[i],
              name=list(config.agent.name),
              rating=ratings[i],
              char=chars[i],
              compile=config.agent.compile,
              batch_steps=config.agent.batch_steps,
              async_inference=config.agent.async_inference,
              override_delay=config.agent.override_delay,
              jax=dataclasses.replace(config.agent.jax),
          ),
          # Learners don't mutate their config, so it's safe to share.
          learner=config.learner,
      )
      for i in range(num_agents)
  ]


class AgentManager(train_two_lib.AgentManager):
  """An agent identified by a 1-based id, which plays the role of the "port"
  in train_two's AgentManager (checkpoint suffix, rng seed, messages)."""

  def __init__(
      self,
      agent_config: train_two_lib.AgentConfig,
      agent_id: int,
      expt_dir: str,
      learner_config: learner_lib.LearnerConfig,
  ):
    super().__init__(
        agent_config=agent_config,
        port=agent_id,
        expt_dir=expt_dir,
        learner_config=learner_config,
    )
    self.id = agent_id
    self.label = f'{agent_id}_{self.character.name.lower()}'

  def set_opponents(self, characters: tp.Iterable[melee.Character]):
    char_name = self.character.name.lower()
    delay = self.learner.policy.delay
    save_name = f'{char_name}_delay_{delay}-{self.id}.pkl'
    save_path = os.path.join(self.expt_dir, save_name)
    if self.save_path is not None:
      assert save_path == self.save_path
    self.save_path = save_path
    self.to_save['opponents'] = sorted(
        set(c.name.lower() for c in characters))


class Source(tp.NamedTuple):
  """Where (part of) an agent's training data comes from."""
  worker: int  # Index into the list of workers.
  port: int  # Key into that worker's trajectories.
  batch_size: int


class WorkerSpec(tp.NamedTuple):
  """A rollout worker for one pair of agents; first == second is self-play."""
  first: int  # Agent indices, playing on ports 1 and 2 respectively.
  second: int
  num_envs: int
  label: str
  build: tp.Callable[[], evaluators.AbstractRolloutWorker]
  # Which agent's variables go to which of the worker's ports.
  port_to_agent: dict[int, int]

  @property
  def is_self_play(self) -> bool:
    return self.first == self.second


@jax.jit
def _concat_batch(
    trajectories: list[learner_lib.FrameSkipTrajectory],
) -> learner_lib.FrameSkipTrajectory:
  """Concatenates frame-skipped trajectories along their batch dimension."""
  return utils.map_nt(
      lambda axis, *xs: jax.tree.map(
          lambda *ys: jnp.concatenate(ys, axis=axis), *xs),
      learner_lib.FrameSkipTrajectory.batch_dims(), *trajectories)


class ExperimentManager:

  def __init__(
      self,
      config: Config,
      agents: list[AgentManager],
      workers: list[WorkerSpec],
      sources: list[list[Source]],  # per agent
      exit_stack: tp.Optional[contextlib.ExitStack] = None,
  ):
    self._config = config
    self._agents = agents
    self._worker_specs = workers
    self._sources = sources
    self._unroll_length = config.actor.rollout_length
    self._num_ppo_batches = config.learner.ppo.num_batches
    self._burnin_steps_after_reset = config.runtime.burnin_steps_after_reset

    self._learner_dtype = jnp.float32
    self._learners = [agent.learner for agent in agents]

    self._hidden_states = [
        learner.initial_state(sum(s.batch_size for s in agent_sources))
        for learner, agent_sources in zip(self._learners, sources)
    ]

    # Fold per-frame rollouts into frame-skipped trajectories. The converters
    # are stateful, so there is one for each source of each agent.
    self._converters = [
        [
            learner_lib.FrameSkipConverter(
                frame_skip=learner.policy.frame_skip,
                batch_shape=(source.batch_size,),
                dummy_sample_outputs=(
                    learner.policy.controller_head.dummy_sample_outputs(
                        [source.batch_size])),
                reward_config=learner._config.reward,
                skip_delay=learner.skip_delay,
            )
            for source in agent_sources
        ]
        for learner, agent_sources in zip(self._learners, sources)
    ]

    # (worker, port) -> (agent index, index into that agent's sources).
    self._piece_index: dict[tuple[int, int], tuple[int, int]] = {}
    for index, agent_sources in enumerate(sources):
      for j, source in enumerate(agent_sources):
        self._piece_index[source.worker, source.port] = (index, j)

    self.update_profiler = utils.Profiler(burnin=0)
    self.learner_profiler = utils.Profiler()
    self.rollout_profiler = utils.Profiler()
    self.reset_profiler = utils.Profiler(burnin=0)

    if not self._config.dolphin.infinite_time:
      logging.info('Finite time mode, disabling env resets')
      self.reset_interval = None
    elif config.actor.use_sim_envs:
      # JaxSimRolloutWorker.reset_env is not implemented.
      logging.info('Sim envs, disabling env resets')
      self.reset_interval = None
    else:
      self.reset_interval = config.runtime.reset_every_n_steps

    with self.reset_profiler:
      self.actors: list[evaluators.AbstractRolloutWorker] = []
      for spec in workers:
        logging.info('Building rollout worker %s', spec.label)
        actor = spec.build()
        actor.start()
        if exit_stack is not None:
          exit_stack.callback(actor.stop)
        self.actors.append(actor)

      self.num_rollouts = 0
      self._burnin_after_reset()

  def _rollout(self) -> tuple[
      list[learner_lib.FrameSkipTrajectory],  # per agent
      list[tp.Mapping[int, evaluators.Trajectory]],  # per worker
      list[dict],  # per worker
  ]:
    self.num_rollouts += 1
    learner_dtype = self._learner_dtype

    # The converted (frame-skipped) pieces of each agent's batch, in the
    # order of its sources.
    pieces: list[list[tp.Optional[learner_lib.FrameSkipTrajectory]]] = [
        [None] * len(agent_sources) for agent_sources in self._sources
    ]
    rollouts: list[tp.Mapping[int, evaluators.Trajectory]] = []
    timings: list[dict] = []

    for worker, actor in enumerate(self.actors):
      trajectories, timing = actor.rollout(self._unroll_length)
      timings.append(timing)

      # Convert each port's trajectory as soon as it is available and drop
      # the actor's sampled outputs, which live on the device (see
      # keep_agent_outputs_on_device), once the converter has copied them.
      # Converting after every worker has rolled out would instead hold two
      # copies of every worker's logits (~1.6 KB per env-frame) at once. The
      # remaining (host) states are what the caller uses for stats.
      stripped: dict[int, evaluators.Trajectory] = {}
      for port, trajectory in trajectories.items():
        index, j = self._piece_index[worker, port]
        fs_trajectory = self._converters[index][j].convert(trajectory)
        agent_dtype = self._agents[index].agent_config.jax.dtype.dtype
        # Agents may run at a lower precision than the learner.
        if agent_dtype != learner_dtype:
          fs_trajectory = fs_trajectory._replace(
              initial_state=run_lib.cast_floats(
                  fs_trajectory.initial_state, dtype=learner_dtype))
        # Transfer the trajectory to the device once; see train_two_lib.
        pieces[index][j] = jax.device_put(fs_trajectory)
        stripped[port] = trajectory._replace(
            actions=None, delayed_actions=None)
      rollouts.append(stripped)

    fs_trajectories: list[learner_lib.FrameSkipTrajectory] = []
    for index in range(len(pieces)):
      # Release each agent's pieces as soon as they are concatenated, rather
      # than holding every agent's pieces until all of the (equally large)
      # concatenations are done.
      agent_pieces = pieces[index]
      pieces[index] = []
      assert all(piece is not None for piece in agent_pieces)
      if len(agent_pieces) == 1:
        fs_trajectories.append(agent_pieces[0])
      else:
        fs_trajectories.append(_concat_batch(agent_pieces))
      del agent_pieces
    return fs_trajectories, rollouts, timings

  def unroll(self):
    """Advance hidden states without training (for burnin)."""
    fs_trajectories, _, _ = self._rollout()
    for i, learner in enumerate(self._learners):
      _, self._hidden_states[i] = learner.unroll(
          fs_trajectories[i], self._hidden_states[i])

  def _burnin_after_reset(self):
    for _ in range(self._burnin_steps_after_reset):
      self.unroll()

  def step(
      self,
      step: int,
  ) -> tuple[list[list[tp.Mapping[int, evaluators.Trajectory]]], dict]:
    """Returns per-batch, per-worker trajectories and metrics."""
    if self.reset_interval and self.num_rollouts >= self.reset_interval:
      logging.info('Resetting environments')
      with self.reset_profiler:
        for actor in self.actors:
          actor.reset_env()
        self.num_rollouts = 0
        self._burnin_after_reset()

    with self.update_profiler:
      # See train_two_lib.AgentManager.policy_variables re: buffer donation.
      variables = [agent.policy_variables() for agent in self._agents]
      for spec, actor in zip(self._worker_specs, self.actors):
        actor.update_variables({
            port: variables[index]
            for port, index in spec.port_to_agent.items()
        })

    with self.rollout_profiler:
      trajectories: list[list[tp.Mapping[int, evaluators.Trajectory]]] = []
      fs_trajectories: list[list[learner_lib.FrameSkipTrajectory]] = [
          [] for _ in self._agents
      ]
      actor_metrics: list[list[dict]] = [[] for _ in self.actors]
      for _ in range(self._num_ppo_batches):
        fs_trajectory, rollouts, timings = self._rollout()
        trajectories.append(rollouts)
        for i, agent_fs_trajectory in enumerate(fs_trajectory):
          fs_trajectories[i].append(agent_fs_trajectory)
        for worker_metrics, timing in zip(actor_metrics, timings):
          timing.pop('completed_games', None)
          worker_metrics.append(timing)

      mean_actor_metrics = {
          spec.label: utils.map_nt(lambda *xs: np.mean(xs), *worker_metrics)
          for spec, worker_metrics in zip(self._worker_specs, actor_metrics)
      }

    with self.learner_profiler:
      # With a single device there's nothing to gain from stepping the
      # learners in parallel (cf. train_two_lib), so just go one by one.
      metrics = {}
      for i, (agent, learner) in enumerate(zip(self._agents, self._learners)):
        self._hidden_states[i], metrics[agent.label] = learner.ppo(
            fs_trajectories[i], self._hidden_states[i], step=step)

      # This blocks on the device, so do it after dispatching every update.
      for agent, learner in zip(self._agents, self._learners):
        learner.check_actor_kl(metrics[agent.label])

    return trajectories, dict(learner=metrics, actor=mean_actor_metrics)


def _cycle_name_pairs(
    names1: list[str], names2: list[str], num_envs: int,
) -> tuple[list[str], list[str]]:
  """Per-env names for both ports, cycling over all name combinations."""
  combinations = itertools.islice(
      itertools.cycle(itertools.product(names1, names2)), num_envs)
  port1_names, port2_names = zip(*combinations)
  return list(port1_names), list(port2_names)


def make_worker_spec(
    config: Config,
    agents: list[AgentManager],
    first: int,
    second: int,
) -> tuple[WorkerSpec, dict[int, list[Source]]]:
  """Builds the rollout worker for a pair of agents.

  Returns the worker's spec and the sources (with a placeholder worker index)
  that it provides to each agent index.
  """
  is_self_play = first == second
  # A cross-play worker only gives each agent one trajectory per env.
  num_envs = config.actor.num_envs * (1 if is_self_play else 2)
  actor_config = dataclasses.replace(config.actor, num_envs=num_envs)

  pair = {1: agents[first], 2: agents[second]}
  if is_self_play:
    label = f'{agents[first].label}_self'
  else:
    label = f'{agents[first].label}_vs_{agents[second].label}'

  port_names = dict(zip(PORTS, _cycle_name_pairs(
      pair[1].name, pair[2].name, num_envs)))

  def get_agent_kwargs() -> dict[int, dict]:
    # Built when the worker is; see AgentManager.agent_kwargs.
    agent_kwargs: dict[int, dict] = {}
    for port, agent in pair.items():
      agent_kwargs[port] = agent.agent_kwargs()
      agent_kwargs[port]['name'] = port_names[port]
    return agent_kwargs

  dolphin_kwargs = dict(
      players={
          port: dolphin_lib.AI(character=agent.character)
          for port, agent in pair.items()
      },
      **config.dolphin.to_kwargs(),
  )

  build: tp.Callable[[], evaluators.AbstractRolloutWorker]

  if config.actor.use_sim_envs:
    if config.actor.async_envs and (
        num_envs % actor_config.get_inner_batch_size()):
      raise ValueError(
          f'{label}: num_envs={num_envs} must be divisible by '
          f'inner_batch_size={actor_config.get_inner_batch_size()} '
          'for sim RL.')

    if is_self_play:
      # The merged agent's trajectory is keyed by its first port.
      sources = {first: [Source(-1, 1, 2 * num_envs)]}
    else:
      sources = {
          first: [Source(-1, 1, num_envs)],
          second: [Source(-1, 2, num_envs)],
      }

    def build() -> evaluators.AbstractRolloutWorker:
      from slippi_ai.sim_env import jax_rollout

      agent_kwargs = get_agent_kwargs()
      rollout_agent_kwargs: dict[int | tuple[int, ...], dict]
      if is_self_play:
        # A single agent batched over both ports.
        merged_kwargs = agent_kwargs[1]
        merged_kwargs['name'] = port_names[1] + port_names[2]
        rollout_agent_kwargs = {PORTS: merged_kwargs}
      else:
        rollout_agent_kwargs = agent_kwargs

      return jax_rollout.JaxSimRolloutWorker(
          agent_kwargs=rollout_agent_kwargs,
          dolphin_kwargs=dolphin_kwargs,
          num_envs=num_envs,
          rollout_length=config.actor.rollout_length,
          use_fake_envs=config.actor.use_fake_envs,
          async_envs=config.actor.async_envs,
          inner_batch_size=actor_config.get_inner_batch_size(),
          # See train_two_lib: with a single ppo batch we consume the data
          # before the next rollout invalidates it.
          copy_data=config.learner.ppo.num_batches > 1,
          keep_agent_outputs_on_device=(
              config.actor.keep_agent_outputs_on_device),
      )
  else:
    # In self-play both ports' trajectories go to the same agent, so there's
    # nothing to gain from swapping ports.
    env_kwargs: dict[str, tp.Any] = dict(
        swap_ports=num_envs > 1 and not is_self_play)
    if config.actor.async_envs:
      env_kwargs.update(
          num_steps=config.actor.num_env_steps,
          inner_batch_size=actor_config.get_inner_batch_size(),
      )

    if is_self_play:
      sources = {first: [Source(-1, port, num_envs) for port in PORTS]}
    else:
      sources = {
          first: [Source(-1, 1, num_envs)],
          second: [Source(-1, 2, num_envs)],
      }

    build = lambda: evaluators.RolloutWorker(
        agent_kwargs=get_agent_kwargs(),
        dolphin_kwargs=dolphin_kwargs,
        env_kwargs=env_kwargs,
        num_envs=num_envs,
        async_envs=config.actor.async_envs,
        use_gpu=config.actor.gpu_inference,
        use_fake_envs=config.actor.use_fake_envs,
    )

  spec = WorkerSpec(
      first=first,
      second=second,
      num_envs=num_envs,
      label=label,
      build=build,
      # The sim worker ignores updates to the second port of a merged agent.
      port_to_agent={1: first, 2: second},
  )
  return spec, sources


def run(
    config: Config,
    agent_specs: tp.Optional[list[AgentSpec]] = None,
):
  """Trains the agents described by config, or by agent_specs if given.

  With agent_specs, config.teachers, config.agent, config.learner and the
  per-agent lists in config are ignored, except that config.learner.ppo and
  config.learner.reward still govern the shared training loop and logging.
  """
  _raise_open_file_limit()
  with contextlib.ExitStack() as exit_stack:
    _run(config, exit_stack, agent_specs)


def _raise_open_file_limit():
  """Raises the soft open-file limit to the hard limit.

  Every matchup's MultiprocessEnv allocates one POSIX shared-memory block per
  array leaf of its observation and action buffers (~160 blocks), and the
  parent keeps a file descriptor open for each one for the lifetime of the
  env. With 55 matchups for 10 agents that is ~9000 descriptors, far above the
  1024 soft limit that login shells typically get, which shows up as
  "OSError: [Errno 24] Too many open files: '/psm_...'". The hard limit is
  usually much higher (1M on systemd systems), and raising the soft limit up
  to it needs no privileges.
  """
  soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
  if soft < hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    logging.info('raised open file limit from %d to %d', soft, hard)


def _run(
    config: Config,
    exit_stack: contextlib.ExitStack,
    agent_specs: tp.Optional[list[AgentSpec]] = None,
):
  tag = config.runtime.tag or jax_train_lib.get_experiment_tag()
  expt_dir = config.runtime.expt_dir
  if expt_dir is None:
    expt_dir = os.path.join(config.runtime.expt_root, tag)
    config.runtime.expt_dir = expt_dir
  os.makedirs(expt_dir, exist_ok=True)
  logging.info('experiment directory: %s', expt_dir)

  if agent_specs is None:
    num_agents = get_num_agents(len(config.teachers), expt_dir)
    agent_specs = get_agent_specs(config, num_agents)
  else:
    num_agents = get_num_agents(len(agent_specs), expt_dir)

  # The loop structure (rollouts per learner step) is shared by all agents.
  for i, spec in enumerate(agent_specs):
    if spec.learner.ppo.num_batches != config.learner.ppo.num_batches:
      raise ValueError(
          f'Agent {i + 1}: learner.ppo.num_batches must match config.learner.')

  agents = [
      AgentManager(
          agent_config=spec.agent,
          agent_id=i + 1,
          expt_dir=expt_dir,
          learner_config=spec.learner,
      )
      for i, spec in enumerate(agent_specs)
  ]

  if config.actor.use_sim_envs:
    for agent in agents:
      batch_steps = agent.agent_config.batch_steps
      delay = agent.learner.policy.delay
      if batch_steps > delay:
        raise ValueError(
            f'Agent {agent.label}: agent.batch_steps={batch_steps} exceeds '
            f'policy delay {delay} for sim RL.')
      if config.actor.rollout_length % max(1, batch_steps):
        raise ValueError(
            'agent.batch_steps must divide rollout_length for sim RL.')

  # One worker for each (unordered) pair of agents.
  workers: list[WorkerSpec] = []
  sources: list[list[Source]] = [[] for _ in agents]
  opponents: list[list[melee.Character]] = [[] for _ in agents]
  for first in range(num_agents):
    for second in range(first, num_agents):
      spec, worker_sources = make_worker_spec(config, agents, first, second)
      for index, agent_sources in worker_sources.items():
        sources[index].extend(
            source._replace(worker=len(workers)) for source in agent_sources)
      workers.append(spec)
      opponents[first].append(agents[second].character)
      opponents[second].append(agents[first].character)

  for agent, agent_opponents, agent_sources in zip(agents, opponents, sources):
    agent.set_opponents(agent_opponents)
    logging.info(
        'Agent %s: learner batch size %d from %d workers',
        agent.label, sum(s.batch_size for s in agent_sources),
        len(set(s.worker for s in agent_sources)))

  experiment_manager = ExperimentManager(
      config=config,
      agents=agents,
      workers=workers,
      sources=sources,
      exit_stack=exit_stack,
  )

  step_profiler = utils.Profiler()
  stalling_threshold = config.learner.reward.stalling_threshold
  env_frames_per_rollout = config.actor.rollout_length * sum(
      spec.num_envs for spec in workers)

  def get_matchup_stats(
      spec: WorkerSpec,
      trajectories: list[evaluators.Trajectory],  # per batch, from port 1
  ) -> dict:
    states: Game = utils.map_nt(
        lambda *xs: np.stack(xs, axis=1), *[t.states for t in trajectories])

    # Note: with sim envs, a self-play trajectory is batched over both ports.
    stats = dict(
        p0=reward.player_stats(
            states.p0, states.p1, states.stage,
            stalling_threshold=stalling_threshold),
    )
    if not spec.is_self_play:
      stats['p1'] = reward.player_stats(
          states.p1, states.p0, states.stage,
          stalling_threshold=stalling_threshold)
      # Positive is good for the first agent.
      stats['ko_diff'] = stats['p1']['deaths'] - stats['p0']['deaths']
    return stats

  def get_log_data(
      all_trajectories: list[list[tp.Mapping[int, evaluators.Trajectory]]],
      metrics: dict,
  ) -> dict:
    step_time = step_profiler.mean_time()
    sps = 1 / step_time
    fps = len(all_trajectories) * env_frames_per_rollout * sps
    mps = fps / MINUTES_PER_FRAME

    timings = dict(
        rollout=experiment_manager.rollout_profiler.mean_time(),
        learner=experiment_manager.learner_profiler.mean_time(),
        reset=experiment_manager.reset_profiler.mean_time(),
        total=step_time,
        sps=sps,
        fps=fps,
        mps=mps,
    )
    timings['actor'] = {
        label: utils.map_nt(lambda x: x * 1000, actor_metrics.pop('timing'))
        for label, actor_metrics in metrics['actor'].items()
    }

    matchups = {
        spec.label: get_matchup_stats(
            spec, [rollouts[i][PORTS[0]] for rollouts in all_trajectories])
        for i, spec in enumerate(workers)
    }

    # Each agent's ko_diff averaged over its cross-play matchups. Every
    # cross-play worker has the same number of envs, so a plain mean over
    # matchups is also the mean over the agent's cross-play trajectories.
    agent_ko_diffs: list[list[float]] = [[] for _ in agents]
    for spec in workers:
      if spec.is_self_play:
        continue
      ko_diff = matchups[spec.label]['ko_diff']
      agent_ko_diffs[spec.first].append(ko_diff)
      agent_ko_diffs[spec.second].append(-ko_diff)
    agent_stats = {
        agent.label: dict(ko_diff=np.mean(ko_diffs))
        for agent, ko_diffs in zip(agents, agent_ko_diffs)
        if ko_diffs  # Empty with a single agent.
    }

    return dict(
        matchups=matchups,
        agents=agent_stats,
        timings=timings,
        actor=metrics['actor'],
        learner=metrics['learner'],
    )

  logger = Logger()
  steps_per_epoch = config.learner.ppo.num_batches

  def flush(step: int):
    total_steps = step * steps_per_epoch
    total_frames = total_steps * env_frames_per_rollout
    extras = dict(
        total_frames=total_frames,
    )

    metrics = logger.flush(total_steps, extras=extras)
    if metrics is None:
      return

    for profiler in [
        step_profiler, experiment_manager.rollout_profiler,
        experiment_manager.learner_profiler, experiment_manager.reset_profiler,
    ]:
      profiler.reset()

    print('\nStep:', step)

    timings: dict = dict(metrics['timings'])
    timings.pop('actor')  # too verbose with many workers
    print(utils.map_nt(lambda v: f'{v:.3f}', timings))

    for label, learner_metrics in metrics['learner'].items():
      post_update = learner_metrics['post_update']
      print(
          f'{label}: '
          f'actor_kl: mean={post_update["actor_kl"]["mean"]:.3g} '
          f'max={post_update["actor_kl"]["max"]:.3g} '
          f'teacher_kl: {post_update["teacher_kl"]:.3g} '
          f'uev: {learner_metrics["value"]["uev"]:.3f}')

    for label, matchup in metrics['matchups'].items():
      if 'ko_diff' in matchup:
        print(f'{label}: ko_diff={matchup["ko_diff"]:.3f}')

    for label, agent_stats in metrics['agents'].items():
      print(f'{label}: mean ko_diff={agent_stats["ko_diff"]:.3f}')

  maybe_flush = utils.Periodically(flush, config.runtime.log_interval)

  def save(step: int):
    for agent in agents:
      agent.save(step)

  maybe_save = utils.Periodically(save, config.runtime.save_interval)

  # All agents share the same step counter.
  steps = set(agent.step for agent in agents)
  if len(steps) != 1:
    raise ValueError(f'Agent checkpoints are at different steps: {steps}')
  step, = steps

  # TODO: flush logger at optimizer/value burnin boundaries
  for agent in agents:
    assert agent.learner._config.optimizer_burnin_epochs == 0
    assert agent.learner._config.value_burnin_epochs == 0

  logging.info('Main training loop')

  while step < config.runtime.max_step:
    with step_profiler:
      trajectories, metrics = experiment_manager.step(step)

    if experiment_manager.learner_profiler.num_calls > 0:
      logger.record(get_log_data(trajectories, metrics))
      maybe_flush(step)

    step += 1
    maybe_save(step)

  save(step)
