"""Benchmarks the eval_two loop: whether agents keep up with Dolphin at 1x.

Runs the scripts/eval_two.py session, but stops after a fixed number of game frames and
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

import json
import time

from absl import app
from absl import flags
import fancyflags as ff
import numpy as np

from slippi_ai import dolphin as dolphin_lib
from slippi_ai import flag_utils, session

PLAYERS = {
    p: ff.DEFINE_dict(f"p{p}", **session.player_flags())
    for p in session.PORTS
}
DOLPHIN = ff.DEFINE_dict('dolphin', **session.dolphin_flags())

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
  config = session.SessionConfig(
      players={port: player.value for port, player in PLAYERS.items()},
      dolphin=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      num_games=1,
  )

  start_frame = WARMUP_FRAMES.value
  end_frame = start_frame + FRAMES.value
  frame_intervals = []
  step_times = []
  last_time = None

  with session.Session(config) as sess:
    for frame in sess.frames():
      # Measured from when the frame arrived, before the agents stepped.
      now = time.perf_counter() - frame.step_time
      measuring = start_frame < frame.gamestate.frame <= end_frame
      if measuring and last_time is not None:
        frame_intervals.append(now - last_time)
      last_time = now

      if measuring:
        step_times.append(frame.step_time)

      if frame.gamestate.frame >= end_frame:
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
