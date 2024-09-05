"""Test character/stage combinations in ffw mode as some cause dolphin errors."""

import dataclasses
import itertools
import json
import time
import typing as tp

from absl import app
from absl import flags
import fancyflags as ff
import tqdm

import melee
from melee import Controller

from slippi_ai import controller_lib
from slippi_ai import flag_utils, envs, utils
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

dolphin_config = dolphin_lib.DolphinConfig(
    headless=True,
    console_timeout=10,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

OUTPUT = flags.DEFINE_string(
    'output', 'test_ffw_results.json', 'output path')

DEBUG = flags.DEFINE_bool('debug', False, 'Enter ipdb on error.')

@dataclasses.dataclass
class EnvConfig:
  rollout_length: int = 300
  num_envs: int = 8
  run_async: bool = True
  inner_batch_size: int = 1

ENV = ff.DEFINE_dict('env', **flag_utils.get_flags_from_dataclass(EnvConfig))


def test(
    chars: dict[int, melee.Character],
    stage: melee.Stage,
    env_config: EnvConfig,
    dolphin_kwargs: dict,
    debug: bool = False,
) -> tp.Union[envs.EnvError, int]:
  env_kwargs = dict(
      num_envs=env_config.num_envs,
      num_retries=0,
  )
  if env_config.run_async:
    env_kwargs.update(
        num_steps=env_config.rollout_length,
        inner_batch_size=env_config.inner_batch_size,
    )

  players = {p: dolphin_lib.AI(character=c) for p, c in chars.items()}
  dolphin_kwargs.update(
      players=players,
      stage=stage,
  )

  if env_config.run_async:
    env_class = envs.AsyncBatchedEnvironmentMP
  else:
    env_class = envs.BatchedEnvironment
  env = env_class(dolphin_kwargs=dolphin_kwargs, **env_kwargs)

  with env.run():
    try:
      controller_inputs: list[dict[int, Controller]] = []
      for _ in range(env_config.rollout_length):
        inputs = {}
        for p in players:
          inputs[p] = utils.batch_nest_nt([
              controller_lib.random_valid_controller()
              for _ in range(env_config.num_envs)])
        env.push(inputs)
        controller_inputs.append(inputs)

      env.pop()  # initial state

      num_wrong = 0

      for i in tqdm.trange(env_config.rollout_length):
        output = env.pop()
        if i < 123:
          continue
        inputs = controller_inputs[i]
        for port, controller in inputs.items():
          process = lambda c: utils.unbatch_nest(controller_lib.to_raw_controller(c))
          unbatched_expected = process(controller)
          unbatched_actual = process(output.gamestates[port].p0.controller)
          for expected, actual in zip(unbatched_expected, unbatched_actual):
            if not actual == expected:
              num_wrong += 1
              if debug:
                print(port, i, utils.check_same_structure(expected, actual, equal=True))
                import ipdb; ipdb.set_trace()
    except envs.EnvError as e:
      return e

  return num_wrong

def main(_):
  env_config = flag_utils.dataclass_from_dict(EnvConfig, ENV.value)
  dolphin_kwargs = dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value)

  from melee import Character
  chars_to_test = [
      Character.FOX,
      Character.FALCO,
      Character.MARTH,
      Character.SHEIK,
      Character.CPTFALCON,
      Character.JIGGLYPUFF,
      Character.PEACH,
      Character.YOSHI,
  ]

  from melee import Stage
  stages_to_test = [
      Stage.YOSHIS_STORY, Stage.BATTLEFIELD, Stage.FINAL_DESTINATION,
      Stage.DREAMLAND, Stage.FOUNTAIN_OF_DREAMS, Stage.POKEMON_STADIUM,
  ]

  results = []

  combinations = list(itertools.product(chars_to_test, chars_to_test, stages_to_test))
  num_errors = 0
  total_num_wrong = 0

  start_time = time.perf_counter()
  for i, (c1, c2, stage) in enumerate(combinations):
    print(c1, c2, stage)
    outcome = test(
        chars={1: c1, 2: c2},
        stage=stage,
        env_config=env_config,
        dolphin_kwargs=dolphin_kwargs,
        debug=DEBUG.value,
    )
    error = None
    num_wrong = None

    if isinstance(outcome, envs.EnvError):
      error = str(outcome)
      num_errors += 1
    elif isinstance(outcome, int):
      num_wrong = outcome
      if num_wrong > 0:
        total_num_wrong += 1

    jsonnable_result = dict(
        chars={1: c1.name, 2: c2.name},
        stage=stage.name,
        error=error,
        num_wrong=num_wrong,
    )
    results.append(jsonnable_result)
    print(outcome)

    elapsed_time = time.perf_counter() - start_time
    n = i + 1
    time_per_item = elapsed_time / n
    time_left = time_per_item * (len(combinations) - n)
    print(f'Estimated time left: {time_left:.0f}')

  print(f'num_errors = {num_errors}, num_wrong = {total_num_wrong}')

  with open(OUTPUT.value, 'w') as f:
    json.dump(results, f)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
