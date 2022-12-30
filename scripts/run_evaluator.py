import pickle

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, dolphin, paths
from slippi_ai import evaluators

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

ROLLOUT_LENGTH = flags.DEFINE_integer(
    'rollout_length', 60 * 60, 'number of steps per rollout')

AGENT = ff.DEFINE_dict('agent',
    path=ff.String(str(paths.DEMO_CHECKPOINT), 'path to agent checkpoint'),
)

def main(_):
  players = {
      1: dolphin.AI(),
      2: dolphin.CPU(),
  }

  env_kwargs = dict(
      players=players,
      **DOLPHIN.value,
  )

  with open(AGENT.value['path'], 'rb') as f:
    state = pickle.load(f)

  config = state['config']
  variables = state['state']['policy']

  evaluator = evaluators.RemoteEvaluator(
      policy_configs={1: config},
      env_kwargs=env_kwargs,
      num_steps_per_rollout=ROLLOUT_LENGTH.value,
  )

  stats = evaluator.rollout({})
  print('uninitialized:', stats)

  stats = evaluator.rollout({1: variables})
  print('initialized:', stats)

if __name__ == '__main__':
  app.run(main)
