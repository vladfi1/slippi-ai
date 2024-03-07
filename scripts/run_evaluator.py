# Make sure not to import things unless we're the main module.
# This allows child processing to avoid importing tensorflow,
# which uses a lot of memory.
if __name__ == '__main__':
  from absl import flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin, utils, evaluators

  DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')
  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')
  SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')

  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')
  RAY_ENVS = flags.DEFINE_boolean('ray_envs', False, 'Use ray environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')

  ASYNC_INFERENCE = flags.DEFINE_boolean('async_inference', False, 'Use async inference.')
  USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')

  AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)

  culprits = []

  def main(_):
    # eval_lib.disable_gpus()

    players = {
        1: dolphin.AI(),
        2: dolphin.AI() if SELF_PLAY.value else dolphin.CPU(),
    }

    env_kwargs = dict(
        players=players,
        **DOLPHIN.value,
    )

    agent_kwargs: dict = AGENT.value.copy()
    state = eval_lib.load_state(
        path=agent_kwargs.pop('path'),
        tag=agent_kwargs.pop('tag'))

    agent_kwargs['state'] = state
    agent_kwargs['batch_steps'] = NUM_AGENT_STEPS.value

    evaluator = evaluators.RemoteEvaluator(
        agent_kwargs={
            port: agent_kwargs for port, player in players.items()
            if isinstance(player, dolphin.AI)},
        env_kwargs=env_kwargs,
        num_envs=NUM_ENVS.value,
        async_envs=ASYNC_ENVS.value,
        ray_envs=RAY_ENVS.value,
        async_inference=ASYNC_INFERENCE.value,
        num_steps_per_rollout=ROLLOUT_LENGTH.value,
        use_gpu=USE_GPU.value,
        extra_env_kwargs=dict(num_steps=NUM_ENV_STEPS.value),
    )

    # burnin
    evaluator.rollout({}, 32)

    timer = utils.Profiler(burnin=0)
    with timer:
      stats, timings = evaluator.rollout({}, ROLLOUT_LENGTH.value)

    print('rewards:', stats)
    print('timings:', timings)

    num_frames = ROLLOUT_LENGTH.value * NUM_ENVS.value
    fps = num_frames / timer.cumtime
    print(f'fps: {fps:.2f}')

  utils.gc_run(main)
