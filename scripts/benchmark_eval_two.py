"""Benchmarks the eval_two loop: whether agents keep up with Dolphin at 1x.

Like scripts/eval_two.py, but stops after a fixed number of game frames and
prints a JSON line with the achieved frame rate, the wall time between frames,
and the time the main loop spends in agent.step. Dolphin waits for agent
inputs (blocking_input), so agents that can't keep up lower the frame rate.

To benchmark an agent against the in-game CPU:

```shell
python scripts/benchmark_eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.type=cpu \
  --p2.ai.path=/path/to/model.onnx \
  --p2.character=fox
```

ONNX agents can be compared on CPU and GPU with --p2.ai.onnx.providers, and
with and without --p2.ai.async_inference.
"""

import contextlib
import json
import os
import time

from absl import app
from absl import flags
import fancyflags as ff
import numpy as np

from slippi_ai import eval_lib, flag_utils, utils
from slippi_ai import dolphin as dolphin_lib

PORTS = (1, 2)

player_flags = utils.map_nt(lambda x: x, eval_lib.PLAYER_FLAGS)
player_flags['ai']['async_inference'] = ff.Boolean(True)

PLAYERS = {p: ff.DEFINE_dict(f"p{p}", **player_flags) for p in PORTS}

dolphin_config = dolphin_lib.DolphinConfig(
    headless=False,
    infinite_time=False,
    online_delay=2,
    emulation_speed=1,
    path=os.environ.get('DOLPHIN_PATH'),
    iso=os.environ.get('ISO_PATH'),
    instant_match_restart=False,
)
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(dolphin_config))

WARMUP_FRAMES = flags.DEFINE_integer(
    'warmup_frames', 60, 'Game frames to skip before measuring.')
FRAMES = flags.DEFINE_integer('frames', 1800, 'Game frames to measure.')
LABEL = flags.DEFINE_string('label', '', 'Label for the results line.')

FRAME_TIME = 1 / 60


def summarize_ms(times: list[float]) -> dict[str, float]:
  ms = np.array(times) * 1000
  return dict(
      mean=round(float(ms.mean()), 2),
      p50=round(float(np.median(ms)), 2),
      p99=round(float(np.percentile(ms, 99)), 2),
      max=round(float(ms.max()), 2),
  )


def main(_):
  with contextlib.ExitStack() as exit_stack:
    _main(exit_stack)


def _main(exit_stack: contextlib.ExitStack):
  eval_lib.disable_gpus()

  players = {
      port: eval_lib.get_player(**player.value)
      for port, player in PLAYERS.items()
  }

  agents: list[eval_lib.Agent] = []

  for port, opponent_port in zip(PORTS, reversed(PORTS)):
    player = players[port]
    if isinstance(player, dolphin_lib.AI):
      agent = eval_lib.build_agent(
          port=port,
          opponent_port=opponent_port,
          console_delay=DOLPHIN.value['online_delay'],
          **PLAYERS[port].value['ai'],
      )
      agent.start()
      agents.append(agent)
      exit_stack.callback(agent.stop)

      eval_lib.update_character(player, agent.config)

  dolphin = dolphin_lib.Dolphin(
      players=players,
      **dolphin_lib.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
  )
  exit_stack.callback(dolphin.stop)

  for agent in agents:
    agent.set_controller(dolphin.controllers[agent._port])

  start_frame = WARMUP_FRAMES.value
  end_frame = start_frame + FRAMES.value
  frame_intervals = []
  step_times = []
  last_time = None

  for gamestate in dolphin.iter_gamestates(skip_menu_frames=False):
    if dolphin_lib.is_menu_state(gamestate):
      if frame_intervals:
        break  # The game ended early.
      continue

    now = time.perf_counter()
    measuring = start_frame < gamestate.frame <= end_frame
    if measuring and last_time is not None:
      frame_intervals.append(now - last_time)
    last_time = now

    for agent in agents:
      agent.step(gamestate)
    if measuring:
      step_times.append(time.perf_counter() - now)

    if gamestate.frame >= end_frame:
      break

  if len(frame_intervals) < FRAMES.value:
    raise RuntimeError(
        f'Only measured {len(frame_intervals)} of {FRAMES.value} frames; '
        'the game ended early.')

  results = dict(
      label=LABEL.value,
      fps=round(len(frame_intervals) / sum(frame_intervals), 2),
      # Frames that took 1.5x longer than they should have.
      slow_frames=int(np.sum(np.array(frame_intervals) > 1.5 * FRAME_TIME)),
      frames=len(frame_intervals),
      frame_interval_ms=summarize_ms(frame_intervals),
      agent_step_ms=summarize_ms(step_times),
  )
  print('RESULTS ' + json.dumps(results), flush=True)


if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
