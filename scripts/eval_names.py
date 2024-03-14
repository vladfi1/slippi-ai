if __name__ == '__main__':
  import itertools

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import eval_lib, dolphin
  from slippi_ai import evaluators, utils

  DOLPHIN_FLAGS = eval_lib.DOLPHIN_FLAGS.copy()
  DOLPHIN_FLAGS['headless'].default = True
  DOLPHIN = ff.DEFINE_dict('dolphin', **DOLPHIN_FLAGS)

  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')
  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', True, 'Use async environments.')
  RAY_ENVS = flags.DEFINE_boolean('ray_envs', False, 'Use ray environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')

  AGENT_FLAGS = eval_lib.AGENT_FLAGS.copy()
  del AGENT_FLAGS['name']
  AGENT_FLAGS['num_names'] = ff.Integer(4, 'Number of names to evaluate.')

  ASYNC_INFERENCE = flags.DEFINE_boolean('async_inference', True, 'Use async inference.')
  USE_GPU = flags.DEFINE_boolean('use_gpu', False, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0, 'Number of agent steps to batch.')

  AGENT = ff.DEFINE_dict('agent', **AGENT_FLAGS)

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')


  def main(_):

    agent_kwargs: dict = AGENT.value.copy()
    state = eval_lib.load_state(
        path=agent_kwargs.pop('path'),
        tag=agent_kwargs.pop('tag'))
    agent_kwargs.update(
        state=state,
        batch_steps=NUM_AGENT_STEPS.value,
        run_on_cpu=not USE_GPU.value,
    )

    num_names: int = agent_kwargs.pop('num_names')
    name_map: dict[str, int] = state['name_map']
    code_to_name = {v: k for k, v in name_map.items()}
    names: list[str] = [code_to_name[i] for i in range(num_names)]
    print('Evaluating names:', names)

    per_name_kwargs: dict[str, dict] = {}
    for name in names:
      per_name_kwargs[name] = agent_kwargs.copy()
      per_name_kwargs[name].update(name=name)

    players = {
        1: dolphin.AI(),
        2: dolphin.AI(),
    }
    env_kwargs = dict(
        players=players,
        **DOLPHIN.value,
    )

    matchups = itertools.combinations(names, 2)

    results = []

    for name1, name2 in list(matchups):
      print(f'Evaluating {name1} vs {name2}')
      evaluator = evaluators.RemoteEvaluator(
          agent_kwargs={1: per_name_kwargs[name1], 2: per_name_kwargs[name2]},
          env_kwargs=env_kwargs,
          num_envs=NUM_ENVS.value,
          async_envs=ASYNC_ENVS.value,
          ray_envs=RAY_ENVS.value,
          async_inference=ASYNC_INFERENCE.value,
          num_steps_per_rollout=ROLLOUT_LENGTH.value,
          use_gpu=USE_GPU.value,
          extra_env_kwargs = dict(num_steps=NUM_ENV_STEPS.value),
      )

      with evaluator.run():
        metrics, timings = evaluator.rollout({}, ROLLOUT_LENGTH.value)
      print(timings)

      total_reward = metrics[1].reward
      num_frames = ROLLOUT_LENGTH.value * NUM_ENVS.value
      mean_reward = total_reward / num_frames
      reward_per_minute = mean_reward * 60 * 60

      print(f'Reward per minute: {reward_per_minute:.2f}')
      results.append((name1, name2, reward_per_minute))

    print('Results:')
    for name1, name2, reward in results:
      print(f'{name1} vs {name2}: {reward:.2f}')

  app.run(main)
