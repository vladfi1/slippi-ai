"""Evaluate two agents against each other and report game outcomes.

This is the library behind scripts/run_evaluator.py. Heavy imports (JAX, the
sim env) happen inside `evaluate` so importing this module stays cheap and
doesn't pull a platform into processes that don't need one.
"""

import dataclasses
import enum
import itertools
import logging
import math
import os
import time
import typing as tp

import fancyflags as ff
import melee
import tree

from slippi_ai import (
    data, dolphin, eval_lib, evaluators, saving, utils)

Port = int
# A (port 1, port 2) character assignment; None keeps the player's character.
CharacterPair = tuple[tp.Optional[melee.Character], tp.Optional[melee.Character]]


def flag_defaults(nest):
  """Replaces every fancyflags Item in a nest with its default value."""
  return tree.map_structure(
      lambda item: item.default if isinstance(item, ff.Item) else item,
      nest)


def batch_agent_flags() -> dict:
  """Agent flags with the compile settings the evaluator wants on by default."""
  agent_flags = utils.deep_copy(eval_lib.BATCH_AGENT_FLAGS)
  agent_flags['tf']['jit_compile'] = ff.Boolean(True)
  agent_flags['jax']['pack_args'] = ff.Boolean(True)
  return agent_flags


def player_flags() -> dict:
  return dict(eval_lib.PLAYER_FLAGS, ai=batch_agent_flags())


def default_player_kwargs(
    path: str, character: tp.Optional[melee.Character] = None) -> dict:
  """A player config dict like the one --player produces, for an AI at path."""
  kwargs = flag_defaults(player_flags())
  kwargs['ai']['path'] = path
  if character is not None:
    kwargs['character'] = character
  return kwargs


@dataclasses.dataclass
class EvaluationConfig:
  """Settings of one evaluation; defaults match scripts/run_evaluator.py."""
  num_envs: int = 1
  rollout_length: int = 60 * 60
  # 0 means one chunk of rollout_length. Smaller chunks cut peak memory.
  chunk_length: int = 0
  # Environment steps to batch in async envs.
  num_env_steps: int = 0
  # Default agent batch_steps; a per-agent ai.batch_steps takes precedence.
  num_agent_steps: int = 0
  # Envs per worker; -1 means num_envs / cpu_count, or 1 if num_envs < cpus.
  inner_batch_size: int = -1
  swap_ports: bool = True  # Dolphin envs only
  fake_envs: bool = False
  sim_envs: bool = False
  async_envs: bool = False
  use_gpu: bool = True
  self_play: bool = False
  # Stop once this many initially active sim games finish; 0 runs the full
  # rollout_length.
  num_games: int = 0
  burnin: bool = False
  quiet: bool = False
  tf_profile: bool = False
  jax_profiler_dir: tp.Optional[str] = None

  def result_affecting(self) -> dict[str, tp.Any]:
    """The fields that change what games get played, for keying evals."""
    d = dataclasses.asdict(self)
    for key in ('quiet', 'burnin', 'tf_profile', 'jax_profiler_dir'):
      del d[key]
    return d


@dataclasses.dataclass
class EvaluationResult:
  config: EvaluationConfig
  total_steps: int
  rewards: dict[Port, float]  # summed reward per port
  kdpm: float                 # port 1's KO differential per game-minute
  completed_games: list[dict]
  timings: dict
  env_fps: float
  player_fps: float
  sps: float
  # (port 1, port 2) character pairs cycled across envs, if overridden.
  character_pairs: list[tuple[str, str]]

  def to_json_dict(self, extra_params: tp.Optional[dict] = None) -> dict:
    params = dataclasses.asdict(self.config)
    params['character_pairs'] = self.character_pairs
    if extra_params:
      params.update(extra_params)
    return dict(
        params=params,
        total_steps=self.total_steps,
        rewards=self.rewards,
        kdpm=self.kdpm,
        env_fps=self.env_fps,
        player_fps=self.player_fps,
        sps=self.sps,
        completed_games=self.completed_games,
    )


def json_default(obj):
  """A `json.dump` default that handles enums and numpy values."""
  if isinstance(obj, enum.Enum):
    return obj.name
  if hasattr(obj, 'item'):  # numpy scalars
    return obj.item()
  if hasattr(obj, 'tolist'):
    return obj.tolist()
  raise TypeError(f'Cannot serialize {type(obj)}')


def summarize_games(games: list[dict]) -> dict[str, tp.Any]:
  """Win/loss/tie counts from port 1's perspective, plus endings and lengths."""
  summary = dict(
      total=len(games),
      wins=sum(game['winner_port'] == 1 for game in games),
      losses=sum(game['winner_port'] == 2 for game in games),
      ties=sum(game['winner_port'] is None for game in games),
      timeouts=sum(game['max_frame_reached'] for game in games),
      stockouts=sum(game['stockout'] for game in games),
  )
  if games:
    lengths = [game['frames'] for game in games]
    stocks = [game['stocks'] for game in games]
    summary.update(
        win_rate=summary['wins'] / len(games),
        avg_frames=sum(lengths) / len(lengths),
        avg_p1_stocks=sum(s[0] for s in stocks) / len(stocks),
        avg_p2_stocks=sum(s[1] for s in stocks) / len(stocks),
    )
  return summary


def format_game_summary(games: list[dict]) -> str:
  s = summarize_games(games)
  if not games:
    return 'completed games: 0'
  lines = [
      'completed games: '
      f'total={s["total"]} wins={s["wins"]} losses={s["losses"]} '
      f'ties={s["ties"]} win_rate={s["win_rate"]:.3f}',
      f'game endings: stockouts={s["stockouts"]} timeouts={s["timeouts"]}',
      'game length: '
      f'avg_frames={s["avg_frames"]:.1f} avg_seconds={s["avg_frames"] / 60:.1f}',
      'final stocks: '
      f'player={s["avg_p1_stocks"]:.2f} opponent={s["avg_p2_stocks"]:.2f}',
  ]
  for stage in sorted({game['stage'] for game in games}):
    stage_games = [game for game in games if game['stage'] == stage]
    ss = summarize_games(stage_games)
    lines.append(
        f'stage {stage}: total={ss["total"]} wins={ss["wins"]} '
        f'losses={ss["losses"]} ties={ss["ties"]} '
        f'win_rate={ss["win_rate"]:.3f} avg_frames={ss["avg_frames"]:.1f}')
  return '\n'.join(lines)


def get_inner_batch_size(config: EvaluationConfig) -> int:
  if config.inner_batch_size != -1:
    return config.inner_batch_size
  cpu_count = os.cpu_count()
  if cpu_count is None:
    raise OSError('Could not determine CPU count for inner_batch_size=-1')
  if config.num_envs < cpu_count:
    return 1
  if config.num_envs % cpu_count != 0:
    raise ValueError(
        f'num_envs={config.num_envs} must be divisible by '
        f'CPU count={cpu_count} for inner_batch_size=-1')
  return config.num_envs // cpu_count


def character_pairs_from_lists(
    characters: tp.Mapping[Port, tp.Sequence[melee.Character]],
) -> list[CharacterPair]:
  """The product of per-port character lists; an empty list means unchanged."""
  per_port = [list(characters.get(port) or [None]) for port in (1, 2)]
  if all(chars == [None] for chars in per_port):
    return []
  return list(itertools.product(*per_port))  # type: ignore[arg-type]


def per_env_dolphin_kwargs(
    dolphin_kwargs: dict,
    players: dict[Port, dolphin.Player],
    character_pairs: tp.Sequence[CharacterPair],
    num_envs: int,
) -> tp.Union[dict, list[dict]]:
  """Cycles envs through (port 1, port 2) character pairs.

  Multi-character agents can then cover several matchups in one run. A None
  in a pair keeps that port's configured character.
  """
  if not character_pairs:
    return dolphin_kwargs
  for pair in character_pairs:
    for port, character in zip((1, 2), pair):
      if character is not None and not isinstance(players[port], dolphin.AI):
        raise ValueError(f'Port {port} is not an AI; cannot set characters.')

  if num_envs % len(character_pairs) != 0:
    logging.warning(
        'num_envs=%d is not divisible by %d character pairs; lanes per pair '
        'will be uneven.', num_envs, len(character_pairs))

  per_env = []
  for i in range(num_envs):
    env_players = {}
    for port, character in zip((1, 2), character_pairs[i % len(character_pairs)]):
      player = players[port]
      if character is not None:
        player = dataclasses.replace(player, character=character)
      env_players[port] = player
    per_env.append(dict(dolphin_kwargs, players=env_players))
  return per_env


def evaluate(
    config: EvaluationConfig,
    player_kwargs: dict[Port, dict],
    dolphin_kwargs: dict,
    character_pairs: tp.Optional[tp.Sequence[CharacterPair]] = None,
) -> EvaluationResult:
  """Runs one evaluation between the players on ports 1 and 2.

  Args:
    config: What and how long to run.
    player_kwargs: Per port, a dict shaped like the --player flag (type,
      character, level, costume, ai=...). With config.self_play, port 1's entry
      is used for both ports.
    dolphin_kwargs: Dolphin/sim config as from DolphinConfig.kwargs_from_flags,
      without `players`.
    character_pairs: Optional (port 1, port 2) character pairs to cycle
      across envs; None in a pair keeps that port's configured character.
  """
  character_pairs = list(character_pairs or [])
  inner_batch_size = get_inner_batch_size(config)
  if config.self_play:
    player_kwargs = {1: player_kwargs[1], 2: player_kwargs[1]}

  agent_kwargs: dict[Port, dict[str, tp.Any]] = {}
  players: dict[Port, dolphin.Player] = {}
  for port, pkwargs in player_kwargs.items():
    player = eval_lib.get_player(**pkwargs)
    players[port] = player
    if isinstance(player, dolphin.AI):
      akwargs: dict = pkwargs['ai'].copy()
      # the evaluator wants the state, not a path
      path = akwargs.pop('path')
      akwargs.update(
          state=saving.load_state_from_disk(path),
          batch_steps=akwargs['batch_steps'] or config.num_agent_steps,
      )
      agent_kwargs[port] = akwargs

  dolphin_kwargs = dict(dolphin_kwargs, players=players)
  dolphin_kwargs = per_env_dolphin_kwargs(
      dolphin_kwargs, players, character_pairs, config.num_envs)

  def resolved(port: Port, character: tp.Optional[melee.Character]) -> str:
    if character is not None:
      return character.name
    player = players[port]
    return player.character.name if isinstance(player, dolphin.AI) else 'HUMAN'
  resolved_pairs = [
      (resolved(1, c1), resolved(2, c2)) for c1, c2 in character_pairs]

  # Completed-game records carry an env_id; the lane's players say who played.
  if isinstance(dolphin_kwargs, dict):
    per_env_players = [dolphin_kwargs['players']] * config.num_envs
  else:
    per_env_players = [kwargs['players'] for kwargs in dolphin_kwargs]
  characters_by_env = [
      tuple(
          p.character.name if isinstance(p, dolphin.AI) else 'HUMAN'
          for p in (env_players[1], env_players[2]))
      for env_players in per_env_players]

  def with_characters(games: list[dict]) -> list[dict]:
    for game in games:
      game['characters'] = characters_by_env[game['env_id']]
    return games

  env_kwargs = dict(swap_ports=config.swap_ports)
  if config.async_envs:
    env_kwargs.update(
        num_steps=config.num_env_steps,
        inner_batch_size=inner_batch_size,
    )

  if config.num_games and not config.sim_envs:
    raise ValueError('num_games currently requires sim_envs.')

  # Every agent's batch_steps must divide the rollout length, so a chunk has
  # to be a multiple of all of them.
  agent_batch_steps = math.lcm(
      *[kwargs['batch_steps'] or 1 for kwargs in agent_kwargs.values()])

  chunk_length = config.chunk_length or config.rollout_length
  if config.rollout_length % chunk_length != 0:
    raise ValueError(
        f'chunk_length ({chunk_length}) must divide '
        f'rollout_length ({config.rollout_length}).')
  if chunk_length % agent_batch_steps != 0:
    raise ValueError(
        f'chunk length ({chunk_length}) must be a multiple of every agent\'s '
        f'batch_steps (lcm={agent_batch_steps}).')

  if config.sim_envs:
    if len(agent_kwargs) != 2:
      raise NotImplementedError(
          'JaxSimRolloutWorker currently only supports 2 agents.')

    from slippi_ai.sim_env import jax_rollout  # pylint: disable=import-outside-toplevel

    sim_agent_kwargs: dict[tp.Union[int, tuple[int, ...]], dict] = {}
    if config.self_play:
      sim_agent_kwargs[(1, 2)] = agent_kwargs[1]
    else:
      sim_agent_kwargs.update(agent_kwargs)

    evaluator = jax_rollout.JaxSimRolloutWorker(
        agent_kwargs=sim_agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.num_envs,
        rollout_length=chunk_length,
        use_fake_envs=config.fake_envs,
        async_envs=config.async_envs,
        inner_batch_size=inner_batch_size,
        # When burnin is enabled, mirror the behavior during RL training.
        keep_agent_outputs_on_device=config.burnin,
    )
  else:
    evaluator = evaluators.Evaluator(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=config.num_envs,
        async_envs=config.async_envs,
        env_kwargs=env_kwargs,
        use_gpu=config.use_gpu,
        use_fake_envs=config.fake_envs,
        use_sim_envs=config.sim_envs,
        damage_ratio=0,
    )

  with evaluator.run():
    if config.burnin:
      burnin_steps = math.ceil(32 / agent_batch_steps) * agent_batch_steps
      if config.sim_envs:
        # JaxSimRolloutWorker only accepts rollouts of its configured length.
        burnin_steps = chunk_length

      print(f'Burning in for {burnin_steps} steps...')
      # Warm up the same code path we time below, not the trajectory one.
      if isinstance(evaluator, evaluators.Evaluator):
        evaluator.rollout(burnin_steps, verbose=not config.quiet)
      else:
        evaluator.rollout_metrics(burnin_steps, verbose=not config.quiet)

    if config.tf_profile:
      import tensorflow as tf  # pylint: disable=import-outside-toplevel
      tf.profiler.experimental.start('tf_profile')

    if config.jax_profiler_dir:
      import jax  # pylint: disable=import-outside-toplevel
      jax.profiler.start_trace(config.jax_profiler_dir)

    cohort = None
    cohort_results = {}
    if config.num_games:
      # Measure a fixed cohort of already-started games. Replacement games
      # after resets are ignored so "100 games" means the first 100 selected
      # games finished, not the first 100 short games to finish.
      active_games = evaluator.active_sim_games()
      if config.num_games > len(active_games):
        raise ValueError(
            'num_games cannot exceed num_envs for unbiased cohort eval.')
      cohort = {
          (game['env_id'], game['episode_id'])
          for game in active_games[:config.num_games]
      }

    timer = utils.Profiler(burnin=0)
    rewards: dict[Port, float] = {}
    completed_games: list[dict] = []
    total_steps = 0

    # A per-rollout bar would restart on every chunk, so when chunking is on
    # we track the whole evaluation with one bar instead. Running to a game
    # cohort has no step target, so that bar is unbounded.
    progress = None
    if not config.quiet and chunk_length != config.rollout_length:
      import tqdm  # pylint: disable=import-outside-toplevel
      progress = tqdm.tqdm(
          total=None if config.num_games else config.rollout_length,
          desc='Rollout', unit='step')

    # Let the worker draw its own per-step bar only when we aren't drawing one.
    verbose = not config.quiet and progress is None

    with timer:
      while True:
        if isinstance(evaluator, evaluators.Evaluator):
          stats, metrics = evaluator.rollout(chunk_length, verbose=verbose)
        else:
          # We only need reward sums, so skip building trajectories: their
          # per-step agent outputs (logits) and encoded states dominate memory.
          stats, metrics = evaluator.rollout_metrics(
              chunk_length, verbose=verbose)
        total_steps += chunk_length
        for port, stat in stats.items():
          rewards[port] = rewards.get(port, 0) + float(stat.reward)
        new_games = with_characters(metrics.get('completed_games', []))
        if cohort is None:
          completed_games.extend(new_games)
        else:
          # Completed-game metadata carries (env_id, episode_id), which lets
          # us keep collecting long games from the original cohort while
          # discarding later episodes from the same env lanes.
          for game in new_games:
            key = (game['env_id'], game['episode_id'])
            if key in cohort:
              cohort_results[key] = game
          completed_games = list(cohort_results.values())
        if progress is not None:
          progress.update(chunk_length)
          if config.num_games:
            progress.set_postfix(
                games=f'{len(completed_games)}/{config.num_games}')
        if config.num_games:
          if len(completed_games) >= config.num_games:
            break
        elif total_steps >= config.rollout_length:
          break

    if progress is not None:
      progress.close()

    if config.tf_profile:
      import tensorflow as tf  # pylint: disable=import-outside-toplevel
      tf.profiler.experimental.stop()

    if config.jax_profiler_dir:
      import jax  # pylint: disable=import-outside-toplevel
      jax.profiler.stop_trace()

  env_frames = config.num_envs * total_steps
  player_frames = env_frames * len(players)
  num_minutes = env_frames / (60 * 60)
  kdpm = rewards[1] / num_minutes

  return EvaluationResult(
      config=config,
      total_steps=total_steps,
      rewards=rewards,
      kdpm=kdpm,
      completed_games=completed_games,
      timings=metrics['timing'],
      env_fps=env_frames / timer.cumtime,
      player_fps=player_frames / timer.cumtime,
      sps=total_steps / timer.cumtime,
      character_pairs=resolved_pairs,
  )


def print_result(result: EvaluationResult):
  print('ko diff per minute:', result.kdpm)
  print(format_game_summary(result.completed_games))
  print('timings:', utils.map_single_structure(
      lambda f: f'{f * 1000:.3f}', result.timings))
  print(
      f'env_fps: {result.env_fps:.2f}, player_fps: {result.player_fps:.2f}, '
      f'sps: {result.sps:.2f}')
