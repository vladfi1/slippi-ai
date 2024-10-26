# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.
import math

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin, utils, evaluators, flag_utils

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
  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')
  INNER_BATCH_SIZE = flags.DEFINE_integer(
      'inner_batch_size', 1, 'Number of environments to run sequentially.')

  USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')
  AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)

  SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')
  OPPONENT = ff.DEFINE_dict('opponent', **eval_lib.PLAYER_FLAGS)

  NUM_WORKERS = flags.DEFINE_integer('num_workers', 0, 'Number of rollout workers.')

  def main(_):
    p1_agent_kwargs: dict = AGENT.value.copy()
    state = eval_lib.load_state(
        path=p1_agent_kwargs.pop('path'),
        tag=p1_agent_kwargs.pop('tag'))
    p1_agent_kwargs.update(
        state=state,
        batch_steps=NUM_AGENT_STEPS.value,
    )

    agent_kwargs = {1: p1_agent_kwargs}

    if SELF_PLAY.value:
      p2 = dolphin.AI()
      agent_kwargs[2] = p1_agent_kwargs
    else:
      p2 = eval_lib.get_player(**OPPONENT.value)

      if isinstance(p2, dolphin.AI):
        p2_agent_kwargs: dict = OPPONENT.value['ai']
        p2_state = eval_lib.load_state(
            path=p2_agent_kwargs.pop('path'),
            tag=p2_agent_kwargs.pop('tag'))
        p2_agent_kwargs.update(
            state=p2_state,
            batch_steps=NUM_AGENT_STEPS.value,
        )
        agent_kwargs[2] = p2_agent_kwargs

    players = {1: dolphin.AI(), 2: p2}
    dolphin_kwargs = dolphin.DolphinConfig.kwargs_from_flags(DOLPHIN.value)
    dolphin_kwargs.update(players=players)

    env_kwargs = {}
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
        damage_ratio=0,
    )

    if NUM_WORKERS.value == 0:
      evaluator = evaluators.Evaluator(**evaluator_kwargs)
    else:
      evaluator = evaluators.RayEvaluator(
          num_workers=NUM_WORKERS.value, **evaluator_kwargs)

    with evaluator.run():
      # burnin
      batch_steps = NUM_AGENT_STEPS.value or 1
      burnin_steps = math.ceil(32 / batch_steps) * batch_steps
      evaluator.rollout(burnin_steps)

      timer = utils.Profiler(burnin=0)
      with timer:
        stats, metrics = evaluator.rollout(ROLLOUT_LENGTH.value)

    num_frames = NUM_ENVS.value * ROLLOUT_LENGTH.value
    num_minutes = num_frames / (60 * 60)
    kdpm = stats[1].reward / num_minutes
    print('ko diff per minute:', kdpm)

    timings = metrics['timing']
    print('timings:', utils.map_single_structure(lambda f: f'{f:.3f}', timings))

    sps = ROLLOUT_LENGTH.value / timer.cumtime
    fps = sps * NUM_ENVS.value
    print(f'fps: {fps:.2f}, sps: {sps:.2f}')

  app.run(main)
