import logging
import unittest

import numpy as np

from slippi_ai import utils, data
from slippi_ai.types import Stick, Controller, Buttons
from slippi_ai.action_space import custom_v1
from slippi_ai.action_space.custom_v1 import (
    S, FloatArray,
)

def normalize_buttons(buttons: Buttons[S]) -> Buttons[S]:
  return buttons._replace(
      X=np.zeros_like(buttons.X),
      Y=buttons.Y | buttons.X,
      L=buttons.L | buttons.R,
      R=np.zeros_like(buttons.R),
  )

def normalize_controller(controller: Controller[S]) -> Controller[S]:
  return controller._replace(
      buttons=normalize_buttons(controller.buttons)
  )

def stick_distance(s1: Stick[S], s2: Stick[S]) -> custom_v1.FloatArray[S]:
  dx = s1.x - s2.x
  dy = s1.y - s2.y
  return np.sqrt(dx * dx + dy * dy)

def controller_diff(c1: Controller[S], c2: Controller[S]) -> Controller[S]:
  main_stick_dist = stick_distance(c1.main_stick, c2.main_stick)
  c_stick_dist = stick_distance(c1.c_stick, c2.c_stick)
  shoulder_dist = np.abs(c1.shoulder - c2.shoulder)

  buttons_equal = utils.map_nt(np.not_equal, c1.buttons, c2.buttons)

  return Controller(
      main_stick=main_stick_dist,
      c_stick=c_stick_dist,
      shoulder=shoulder_dist,
      buttons=buttons_equal,
  )

def controller_distance(
    c1: Controller[S], c2: Controller[S],
    main_stick_weight: float = 1.0,
    c_stick_weight: float = 0.1,
    shoulder_weight: float = 1.0,
    button_weight: float = 1.0,
) -> FloatArray[S]:
  diff = controller_diff(c1, c2)

  return sum([
      diff.main_stick * main_stick_weight,
      diff.c_stick * c_stick_weight,
      diff.shoulder * shoulder_weight,
      sum(diff.buttons) * button_weight,
  ])

class TestCustomV1(unittest.TestCase):

  def test_roundtrip_random_labels(self):
    bucketer = custom_v1.Config.default().create_bucketer()
    button_size, main_stick_size = bucketer.axis_sizes

    batch_size = 1000
    generator = np.random.default_rng()
    button_labels = generator.integers(0, button_size, size=(batch_size,), dtype=np.uint16)
    main_stick_labels = generator.integers(0, main_stick_size, size=(batch_size,), dtype=np.uint16)

    decoded = bucketer.decode(button_labels, main_stick_labels)
    reencoded = bucketer.bucket(decoded)

    np.testing.assert_array_equal(button_labels, reencoded[0])
    np.testing.assert_array_equal(main_stick_labels, reencoded[1])

  def test_controller_distance(self):
    toy_data = data.toy_data_source(batch_size=1, unroll_length=3000)
    batch, _ = next(toy_data)
    game = batch.frames.state_action.state
    c0 = game.p0.controller
    d0 = controller_distance(c0, c0)
    np.testing.assert_array_equal(d0, 0.0)

  def test_distance_toy_data(self):
    bucketer = custom_v1.Config.default().create_bucketer()

    def get_distance(c: Controller[S]) -> FloatArray[S]:
      label = bucketer.bucket(c)
      decoded = bucketer.decode(*label)
      return controller_distance(normalize_controller(c), decoded)

    toy_data = data.toy_data_source(batch_size=1, unroll_length=3000)
    batch, _ = next(toy_data)
    game = batch.frames.state_action.state
    game = utils.map_nt(lambda arr: arr[0], game)  # Unbatch

    for c_old in [game.p0.controller, game.p1.controller]:
      c_old = game.p0.controller
      c_new = bucketer.decode(*bucketer.bucket(c_old))

      distance = get_distance(c_old)
      assert distance.mean() < 1e-2

      thresholds = [
          (0.1, 0.01),  # 1% of frames have distance > 0.1
      ]

      for threshold, fraction in thresholds:
        above_threshold = np.mean(distance > threshold)
        logging.info(f"Distance > {threshold}: {above_threshold:.4f}")
        assert above_threshold < fraction

      worst_idx = np.argmax(distance)
      logging.info(f"Worst frame index: {worst_idx}")
      logging.info(f"Worst frame distance: {distance[worst_idx]}")

      c_old_worst = utils.map_nt(lambda arr: arr[worst_idx], c_old)
      c_new_worst = utils.map_nt(lambda arr: arr[worst_idx], c_new)
      diff_worst = controller_diff(normalize_controller(c_old_worst), c_new_worst)
      logging.info(f"Worst frame old controller: {c_old_worst}")
      logging.info(f"Worst frame new controller: {c_new_worst}")
      logging.info(f"Worst frame diff: {diff_worst}")

if __name__ == "__main__":
  unittest.main()
