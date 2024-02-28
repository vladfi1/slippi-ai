from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, dolphin, utils, evaluators

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

ROLLOUT_LENGTH = flags.DEFINE_integer(
    'rollout_length', 60 * 60, 'number of steps per rollout')
NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')
SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')
ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')

AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)

def main(_):
  eval_lib.disable_gpus()

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

  evaluator = evaluators.RemoteEvaluator(
      agent_kwargs={
          port: agent_kwargs for port, player in players.items()
          if isinstance(player, dolphin.AI)},
      env_kwargs=env_kwargs,
      num_envs=NUM_ENVS.value,
      async_envs=ASYNC_ENVS.value,
      num_steps_per_rollout=ROLLOUT_LENGTH.value,
  )

  timer = utils.Profiler()
  with timer:
    stats, timings = evaluator.rollout({})

  print('rewards:', stats)
  print('timings:', timings)

  num_frames = ROLLOUT_LENGTH.value * NUM_ENVS.value
  fps = num_frames / timer.cumtime
  print(f'fps: {fps:.2f}')

if __name__ == '__main__':
  app.run(main)
