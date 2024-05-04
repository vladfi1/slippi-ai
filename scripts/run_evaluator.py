# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.
if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin, utils, evaluators

  DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')
  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')

  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')
  INNER_BATCH_SIZE = flags.DEFINE_integer(
      'inner_batch_size', 1, 'Number of environments to run sequentially.')

  SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')
  ASYNC_INFERENCE = flags.DEFINE_boolean('async_inference', False, 'Use async inference.')
  USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')
  AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)

  NUM_WORKERS = flags.DEFINE_integer('num_workers', 0, 'Number of rollout workers.')

  def main(_):

    players = {
        1: dolphin.AI(),
        2: dolphin.AI() if SELF_PLAY.value else dolphin.CPU(),
    }

    dolphin_kwargs = dict(
        players=players,
        **DOLPHIN.value,
    )

    agent_kwargs: dict = AGENT.value.copy()
    state = eval_lib.load_state(
        path=agent_kwargs.pop('path'),
        tag=agent_kwargs.pop('tag'))
    agent_kwargs.update(
        state=state,
        batch_steps=NUM_AGENT_STEPS.value,
    )

    env_kwargs = {}
    if ASYNC_ENVS.value:
      env_kwargs.update(
          num_steps=NUM_ENV_STEPS.value,
          inner_batch_size=INNER_BATCH_SIZE.value,
      )

    evaluator_kwargs = dict(
        agent_kwargs={
            port: agent_kwargs for port, player in players.items()
            if isinstance(player, dolphin.AI)},
        dolphin_kwargs=dolphin_kwargs,
        num_envs=NUM_ENVS.value,
        async_envs=ASYNC_ENVS.value,
        env_kwargs=env_kwargs,
        async_inference=ASYNC_INFERENCE.value,
        use_gpu=USE_GPU.value,
    )

    if NUM_WORKERS.value == 0:
      evaluator = evaluators.Evaluator(**evaluator_kwargs)
    else:
      evaluator = evaluators.RayEvaluator(
          num_workers=NUM_WORKERS.value, **evaluator_kwargs)

    with evaluator.run():
      # burnin
      evaluator.rollout(32)
      # import time; time.sleep(1000)

      timer = utils.Profiler(burnin=0)
      with timer:
        stats, timings = evaluator.rollout(ROLLOUT_LENGTH.value)

    print('rewards:', stats)
    print('timings:', utils.map_single_structure(lambda f: f'{f:.3f}', timings))

    sps = ROLLOUT_LENGTH.value / timer.cumtime
    fps = sps * NUM_ENVS.value
    print(f'fps: {fps:.2f}, sps: {sps:.2f}')

  app.run(main)
