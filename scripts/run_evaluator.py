# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.
import math

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin, utils, evaluators, flag_utils, saving

  default_dolphin_config = dolphin.DolphinConfig(
      infinite_time=False,
      headless=True,
  )
  DOLPHIN = ff.DEFINE_dict(
      'dolphin', **flag_utils.get_flags_from_default(default_dolphin_config))

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')
  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')

  FAKE_ENVS = flags.DEFINE_boolean('fake_envs', False, 'Use fake environments.')
  SIM_ENVS = flags.DEFINE_boolean('sim_envs', False, 'Use melee-sim-light environments.')
  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')
  INNER_BATCH_SIZE = flags.DEFINE_integer(
      'inner_batch_size', 1, 'Number of environments to run sequentially.')
  SWAP_PORTS = flags.DEFINE_boolean('swap_ports', True, 'Swap half of env ports.')

  USE_GPU = flags.DEFINE_boolean('use_gpu', True, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')

  agent_flags = utils.deep_copy(eval_lib.BATCH_AGENT_FLAGS)
  agent_flags['tf']['jit_compile'] = ff.Boolean(True)

  player_flags = dict(eval_lib.PLAYER_FLAGS, ai=agent_flags)
  PLAYER = ff.DEFINE_dict('player', **player_flags)

  SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')
  OPPONENT = ff.DEFINE_dict('opponent', **player_flags)

  TF_PROFILE = flags.DEFINE_boolean('tf_profile', False, 'Enable TF profiler.')
  JAX_PROFILER_DIR = flags.DEFINE_string('jax_profiler_dir', None, 'Directory for JAX profiler traces.')
  NUM_GAMES = flags.DEFINE_integer(
      'num_games', 0, 'Stop after this many initially active sim games finish.')

  def print_game_summary(games: list[dict]):
    # Sim envs expose completed-game records directly, so this summary reports
    # game-level outcomes instead of inferring strength from rollout reward.
    if not games:
      print('completed games: 0')
      return

    wins = sum(game['winner_port'] == 1 for game in games)
    losses = sum(game['winner_port'] == 2 for game in games)
    ties = sum(game['winner_port'] is None for game in games)
    timeouts = sum(game['max_frame_reached'] for game in games)
    stockouts = sum(game['stockout'] for game in games)
    lengths = [game['frames'] for game in games]
    stocks = [game['stocks'] for game in games]
    print(
        'completed games: '
        f'total={len(games)} wins={wins} losses={losses} ties={ties} '
        f'win_rate={wins / len(games):.3f}')
    print(
        'game endings: '
        f'stockouts={stockouts} timeouts={timeouts}')
    print(
        'game length: '
        f'avg_frames={sum(lengths) / len(lengths):.1f} '
        f'avg_seconds={sum(lengths) / len(lengths) / 60:.1f}')
    print(
        'final stocks: '
        f'player={sum(s[0] for s in stocks) / len(stocks):.2f} '
        f'opponent={sum(s[1] for s in stocks) / len(stocks):.2f}')
    stages = sorted({game['stage'] for game in games})
    for stage in stages:
      stage_games = [game for game in games if game['stage'] == stage]
      stage_wins = sum(game['winner_port'] == 1 for game in stage_games)
      stage_losses = sum(game['winner_port'] == 2 for game in stage_games)
      stage_ties = sum(game['winner_port'] is None for game in stage_games)
      stage_lengths = [game['frames'] for game in stage_games]
      print(
          f'stage {stage}: total={len(stage_games)} '
          f'wins={stage_wins} losses={stage_losses} ties={stage_ties} '
          f'win_rate={stage_wins / len(stage_games):.3f} '
          f'avg_frames={sum(stage_lengths) / len(stage_lengths):.1f}')

  def main(_):
    player_kwargs = {
        1: PLAYER.value,
        2: PLAYER.value if SELF_PLAY.value else OPPONENT.value,
    }
    agent_kwargs = {}
    players = {}
    for port, pkwargs in player_kwargs.items():
      player = eval_lib.get_player(**pkwargs)
      players[port] = player
      if isinstance(player, dolphin.AI):
        akwargs: dict = pkwargs['ai'].copy()
        # the evaluator wants the state, not a path
        path = akwargs.pop('path')
        akwargs.update(
            state=saving.load_state_from_disk(path),
            batch_steps=NUM_AGENT_STEPS.value,
        )
        agent_kwargs[port] = akwargs

    dolphin_kwargs = dolphin.DolphinConfig.kwargs_from_flags(DOLPHIN.value)
    dolphin_kwargs.update(players=players)

    env_kwargs = dict(
        swap_ports=SWAP_PORTS.value,
    )
    if ASYNC_ENVS.value:
      env_kwargs.update(
          num_steps=NUM_ENV_STEPS.value,
          inner_batch_size=INNER_BATCH_SIZE.value,
      )

    evaluator_kwargs = dict(
        agent_kwargs=agent_kwargs,
        dolphin_kwargs=dolphin_kwargs,
        num_envs=NUM_ENVS.value,
        async_envs=ASYNC_ENVS.value,
        env_kwargs=env_kwargs,
        use_gpu=USE_GPU.value,
        use_fake_envs=FAKE_ENVS.value,
        use_sim_envs=SIM_ENVS.value,
        damage_ratio=0,
    )
    if NUM_GAMES.value and not SIM_ENVS.value:
      raise ValueError('--num_games currently requires --sim_envs.')

    evaluator = evaluators.Evaluator(**evaluator_kwargs)

    with evaluator.run():
      # burnin
      batch_steps = NUM_AGENT_STEPS.value or 1
      burnin_steps = math.ceil(32 / batch_steps) * batch_steps
      evaluator.rollout(burnin_steps)

      if TF_PROFILE.value:
        import tensorflow as tf
        tf.profiler.experimental.start('tf_profile')

      if JAX_PROFILER_DIR.value:
        import jax
        jax.profiler.start_trace(JAX_PROFILER_DIR.value)

      cohort = None
      cohort_results = {}
      if NUM_GAMES.value:
        # Measure a fixed cohort of already-started games. Replacement games
        # after resets are ignored so "100 games" means the first 100 selected
        # games finished, not the first 100 short games to finish.
        active_games = evaluator.active_sim_games()
        if NUM_GAMES.value > len(active_games):
          raise ValueError(
              '--num_games cannot exceed --num_envs for unbiased cohort eval.')
        cohort = {
            (game['env_id'], game['episode_id'])
            for game in active_games[:NUM_GAMES.value]
        }

      timer = utils.Profiler(burnin=0)
      rewards = {}
      completed_games = []
      total_steps = 0
      with timer:
        while True:
          stats, metrics = evaluator.rollout(ROLLOUT_LENGTH.value, verbose=True)
          total_steps += ROLLOUT_LENGTH.value
          for port, stat in stats.items():
            rewards[port] = rewards.get(port, 0) + stat.reward
          if cohort is None:
            completed_games.extend(metrics.get('completed_games', []))
          else:
            # Completed-game metadata carries (env_id, episode_id), which lets
            # us keep collecting long games from the original cohort while
            # discarding later episodes from the same env lanes.
            for game in metrics.get('completed_games', []):
              key = (game['env_id'], game['episode_id'])
              if key in cohort:
                cohort_results[key] = game
            completed_games = list(cohort_results.values())
          if not NUM_GAMES.value or len(completed_games) >= NUM_GAMES.value:
            break

      if TF_PROFILE.value:
        import tensorflow as tf
        tf.profiler.experimental.stop()

      if JAX_PROFILER_DIR.value:
        import jax
        jax.profiler.stop_trace()

    env_frames = NUM_ENVS.value * total_steps
    player_frames = env_frames * len(players)
    num_minutes = env_frames / (60 * 60)
    kdpm = rewards[1] / num_minutes
    print('ko diff per minute:', kdpm)
    print_game_summary(completed_games)

    timings = metrics['timing']
    print('timings:', utils.map_single_structure(lambda f: f'{f * 1000:.3f}', timings))

    sps = total_steps / timer.cumtime
    env_fps = env_frames / timer.cumtime
    player_fps = player_frames / timer.cumtime
    print(
        f'env_fps: {env_fps:.2f}, player_fps: {player_fps:.2f}, '
        f'sps: {sps:.2f}')

  app.run(main)
