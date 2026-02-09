"""Measure bucketing distance on real replay data.

Samples replays, buckets controller inputs through custom_v1, and measures
the reconstruction distance to quantify information loss.
"""

import logging
import os

from absl import app, flags
import numpy as np

from slippi_ai import utils
from slippi_ai.data import read_table
from slippi_ai.types import Stick, Controller, Buttons
from slippi_ai.action_space import custom_v1
from slippi_ai.action_space.custom_v1 import S, FloatArray
from slippi_ai.action_space.clustering import sample_replays_by_character

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', os.environ.get('DATA_DIR'), 'Directory containing replay data files')
flags.DEFINE_string('meta_path', os.environ.get('META_PATH'), 'Path to metadata JSON file')
flags.DEFINE_integer('samples_per_character', 50, 'Number of replays to sample per character')
flags.DEFINE_integer('max_frames_per_game', 1000, 'Max frames to sample per game (0 = all)')
flags.DEFINE_integer('seed', 42, 'Random seed for sampling')
flags.DEFINE_float('main_stick_weight', 1.0, 'Weight for main stick distance')
flags.DEFINE_float('c_stick_weight', 0.1, 'Weight for c-stick distance')
flags.DEFINE_float('shoulder_weight', 1.0, 'Weight for shoulder distance')
flags.DEFINE_float('button_weight', 1.0, 'Weight for button distance')


def normalize_controller(controller: Controller[S]) -> Controller[S]:
  buttons = controller.buttons._replace(
      X=np.zeros_like(controller.buttons.X),
      Y=controller.buttons.Y | controller.buttons.X,
      L=controller.buttons.L | controller.buttons.R,
      R=np.zeros_like(controller.buttons.R),
      D_UP=np.zeros_like(controller.buttons.D_UP),
  )
  return controller._replace(buttons=buttons)


def stick_distance(s1: Stick[S], s2: Stick[S]) -> FloatArray[S]:
  dx = s1.x - s2.x
  dy = s1.y - s2.y
  return np.sqrt(dx * dx + dy * dy)


def controller_diff(c1: Controller[S], c2: Controller[S]) -> Controller[S]:
  return Controller(
      main_stick=stick_distance(c1.main_stick, c2.main_stick),
      c_stick=stick_distance(c1.c_stick, c2.c_stick),
      shoulder=np.abs(c1.shoulder - c2.shoulder),
      buttons=utils.map_nt(np.not_equal, c1.buttons, c2.buttons),
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


def extract_controllers(
    sampled_meta: list,
    data_dir: str,
    max_frames_per_game: int,
    seed: int,
) -> Controller:
  """Extract controller data from sampled replays as Controller NamedTuples."""
  rng = np.random.RandomState(seed)
  all_controllers: list[Controller] = []

  for row in sampled_meta:
    path = os.path.join(data_dir, row['slp_md5'])
    try:
      game = read_table(path, compressed=True)
    except Exception as e:
      logging.warning(f"Failed to read {path}: {e}")
      continue

    for player in [game.p0, game.p1]:
      ctrl = player.controller
      n_frames = len(ctrl.buttons.A)

      if max_frames_per_game > 0 and n_frames > max_frames_per_game:
        indices = rng.choice(n_frames, max_frames_per_game, replace=False)
      else:
        indices = np.arange(n_frames)

      sampled = utils.map_nt(lambda arr: arr[indices], ctrl)
      all_controllers.append(sampled)

  return utils.map_nt(lambda *arrs: np.concatenate(arrs, axis=0), *all_controllers)


def main(_):
  if not FLAGS.data_dir or not FLAGS.meta_path:
    raise ValueError("--data_dir and --meta_path must be set")

  print(f"Data dir: {FLAGS.data_dir}")
  print(f"Meta path: {FLAGS.meta_path}")

  sampled = sample_replays_by_character(
      FLAGS.meta_path,
      FLAGS.data_dir,
      samples_per_character=FLAGS.samples_per_character,
      seed=FLAGS.seed,
  )

  print(f"\nExtracting controller data from {len(sampled)} replays...")
  controller = extract_controllers(
      sampled,
      FLAGS.data_dir,
      max_frames_per_game=FLAGS.max_frames_per_game,
      seed=FLAGS.seed,
  )
  n_frames = len(controller.buttons.A)
  print(f"Extracted {n_frames:,} frames")

  bucketer = custom_v1.Config.default().create_bucketer()

  labels = bucketer.bucket(controller)
  decoded = bucketer.decode(*labels)
  normalized = normalize_controller(controller)

  distance = controller_distance(
      normalized, decoded,
      main_stick_weight=FLAGS.main_stick_weight,
      c_stick_weight=FLAGS.c_stick_weight,
      shoulder_weight=FLAGS.shoulder_weight,
      button_weight=FLAGS.button_weight,
  )

  print(f"\n{'='*60}")
  print(f"BUCKETING DISTANCE ANALYSIS")
  print(f"{'='*60}")
  print(f"Mean distance: {distance.mean():.6f}")
  print(f"Median distance: {np.median(distance):.6f}")
  print(f"Max distance: {distance.max():.6f}")
  print(f"Std distance: {distance.std():.6f}")

  for threshold in [0.01, 0.05, 0.1, 0.2, 0.5]:
    frac = np.mean(distance > threshold)
    print(f"Fraction > {threshold}: {frac:.6f} ({frac*100:.3f}%)")

  # Per-component breakdown
  diff = controller_diff(normalized, decoded)
  print(f"\nPer-component mean distances:")
  print(f"  Main stick: {diff.main_stick.mean():.6f}")
  print(f"  C-stick:    {diff.c_stick.mean():.6f}")
  print(f"  Shoulder:   {diff.shoulder.mean():.6f}")
  for name in Buttons._fields:
    btn_diff = getattr(diff.buttons, name).mean()
    if btn_diff > 0:
      print(f"  Button {name:>4}: {btn_diff:.6f}")

  # Worst frames
  worst_indices = np.argsort(distance)[-5:][::-1]
  print(f"\nWorst 5 frames:")
  for idx in worst_indices:
    old = utils.map_nt(lambda arr: arr[idx], normalized)
    new = utils.map_nt(lambda arr: arr[idx], decoded)
    d = controller_diff(old, new)
    print(f"  Frame {idx}: distance={distance[idx]:.4f}")
    print(f"    main_stick: {d.main_stick:.4f}")
    print(f"    c_stick:    {d.c_stick:.4f}")
    print(f"    shoulder:   {d.shoulder:.4f}")
    btn_diffs = {name: getattr(d.buttons, name) for name in Buttons._fields if getattr(d.buttons, name)}
    if btn_diffs:
      print(f"    buttons:    {btn_diffs}")


if __name__ == '__main__':
  app.run(main)
