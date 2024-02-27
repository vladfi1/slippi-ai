from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import eval_lib, dolphin
from slippi_ai import evaluators

DOLPHIN = ff.DEFINE_dict('dolphin', **eval_lib.DOLPHIN_FLAGS)

ROLLOUT_LENGTH = flags.DEFINE_integer(
    'rollout_length', 60 * 60, 'number of steps per rollout')

AGENT = ff.DEFINE_dict('agent', **eval_lib.AGENT_FLAGS)

def main(_):
  eval_lib.disable_gpus()

  players = {
      1: dolphin.AI(),
      2: dolphin.CPU(),
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

  port = 1
  evaluator = evaluators.RemoteEvaluator(
      agent_kwargs={port: agent_kwargs},
      dolphin_kwargs=env_kwargs,
      num_steps_per_rollout=ROLLOUT_LENGTH.value,
  )

  stats = evaluator.rollout({})
  print(stats)

if __name__ == '__main__':
  app.run(main)
