import pickle

from absl import app
from absl import flags
import fancyflags as ff

import ray

from slippi_ai import eval_lib, dolphin, paths
from slippi_ai import evaluators

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

ROLLOUT_LENGTH = flags.DEFINE_integer(
    'rollout_length', 60 * 60, 'number of steps per rollout')

AGENT = ff.DEFINE_dict('agent',
    path=ff.String(str(paths.DEMO_CHECKPOINT), 'path to agent checkpoint'),
)

USE_RAY = flags.DEFINE_bool('use_ray', False, 'run "remotely" with ray')


def run(evaluator_kwargs, variables):
  evaluator = evaluators.SerializableRolloutWorker(**evaluator_kwargs)

  stats = evaluator.rollout({})
  print('uninitialized:', stats)

  stats = evaluator.rollout({1: variables})
  print('initialized:', stats)

  evaluator.stop()


def run_ray(evaluator_kwargs, variables):
  evaluator = evaluators.RayRolloutWorker.remote(**evaluator_kwargs)

  # fire off two asynchronous evaluations
  pre = evaluator.rollout.remote({})
  post = evaluator.rollout.remote({1: variables})

  print('Evaluations started.')
  pre, post = ray.get([pre, post])

  print('uninitialized:', pre)
  print('initialized:', post)

  ray.wait([evaluator.stop.remote()])


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

  evaluator_kwargs = dict(
      policy_configs={1: config},
      env_kwargs=env_kwargs,
      num_steps_per_rollout=ROLLOUT_LENGTH.value,
  )

  run_fn = run_ray if USE_RAY.value else run
  run_fn(evaluator_kwargs, variables)

if __name__ == '__main__':
  app.run(main)
