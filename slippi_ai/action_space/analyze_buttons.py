"""Analyze button relationships and equivalences in controller data.

This script checks:
1. Whether X and Y buttons are used equivalently
2. Whether L and R buttons are used equivalently
3. Whether L/R button correlates with full shoulder press
4. Whether Z implies A+L/R (grab mechanics)
"""

import argparse
import os

import numpy as np

from slippi_ai.action_space.clustering import (
    sample_replays_by_character,
    extract_controllers_from_replays,
)


def analyze_button_equivalences(data: dict) -> dict:
  """Analyze whether X=Y and L=R always hold in raw data.

  Args:
      data: Raw (unnormalized) controller data from extract_controllers_from_replays.

  Returns:
      Dict with analysis results.
  """
  buttons = data['buttons']
  # Button order: A, B, X, Y, Z, L, R, D_UP
  X = buttons[:, 2].astype(bool)
  Y = buttons[:, 3].astype(bool)
  L = buttons[:, 5].astype(bool)
  R = buttons[:, 6].astype(bool)

  n_frames = len(buttons)

  # Check X vs Y
  x_only = (X & ~Y).sum()
  y_only = (Y & ~X).sum()
  both_xy = (X & Y).sum()
  neither_xy = (~X & ~Y).sum()

  # Check L vs R
  l_only = (L & ~R).sum()
  r_only = (R & ~L).sum()
  both_lr = (L & R).sum()
  neither_lr = (~L & ~R).sum()

  print("\n" + "="*60)
  print("BUTTON EQUIVALENCE ANALYSIS")
  print("="*60)

  print(f"\nX vs Y (both are jump):")
  print(f"  X only (no Y): {x_only:,} ({x_only/n_frames*100:.3f}%)")
  print(f"  Y only (no X): {y_only:,} ({y_only/n_frames*100:.3f}%)")
  print(f"  Both X and Y:  {both_xy:,} ({both_xy/n_frames*100:.3f}%)")
  print(f"  Neither:       {neither_xy:,} ({neither_xy/n_frames*100:.3f}%)")
  print(
      f"  -> X=Y always holds: {x_only == 0 and y_only == 0 and both_xy == 0}")

  print(f"\nL vs R (both are shield):")
  print(f"  L only (no R): {l_only:,} ({l_only/n_frames*100:.3f}%)")
  print(f"  R only (no L): {r_only:,} ({r_only/n_frames*100:.3f}%)")
  print(f"  Both L and R:  {both_lr:,} ({both_lr/n_frames*100:.3f}%)")
  print(f"  Neither:       {neither_lr:,} ({neither_lr/n_frames*100:.3f}%)")
  print(
      f"  -> L=R always holds: {l_only == 0 and r_only == 0 and both_lr == 0}")

  return {
      'x_only': x_only,
      'y_only': y_only,
      'both_xy': both_xy,
      'l_only': l_only,
      'r_only': r_only,
      'both_lr': both_lr,
      'x_equals_y': x_only == 0 and y_only == 0 and both_xy == 0,
      'l_equals_r': l_only == 0 and r_only == 0 and both_lr == 0,
  }


def analyze_shoulder_button_relationship(data: dict) -> dict:
  """Analyze if L/R button always coincides with full shoulder press.

  Args:
      data: Raw (unnormalized) controller data.

  Returns:
      Dict with analysis results.
  """
  buttons = data['buttons']
  shoulder = data['shoulder']

  # Button order: A, B, X, Y, Z, L, R, D_UP
  L_btn = buttons[:, 5].astype(bool)
  R_btn = buttons[:, 6].astype(bool)
  LR_btn = L_btn | R_btn

  n_frames = len(buttons)

  # Define "full press" threshold
  full_press = shoulder > 0.9

  # Check relationship
  btn_and_full = (LR_btn & full_press).sum()
  btn_no_full = (LR_btn & ~full_press).sum()
  full_no_btn = (full_press & ~LR_btn).sum()
  neither = (~LR_btn & ~full_press).sum()

  print("\n" + "="*60)
  print("L/R BUTTON vs SHOULDER PRESS ANALYSIS")
  print("="*60)

  print(f"\nL/R button vs full shoulder press (>0.9):")
  print(
      f"  L/R btn AND full press:  {btn_and_full:,} ({btn_and_full/n_frames*100:.3f}%)")
  print(
      f"  L/R btn, NO full press:  {btn_no_full:,} ({btn_no_full/n_frames*100:.3f}%)")
  print(
      f"  Full press, NO L/R btn:  {full_no_btn:,} ({full_no_btn/n_frames*100:.3f}%)")
  print(
      f"  Neither:                 {neither:,} ({neither/n_frames*100:.3f}%)")
  print(f"  -> L/R btn = full press: {btn_no_full == 0 and full_no_btn == 0}")

  # Also check light press
  light_press = (shoulder > 0.3) & (shoulder <= 0.9)
  light_with_btn = (light_press & LR_btn).sum()
  light_no_btn = (light_press & ~LR_btn).sum()

  print(f"\nLight shoulder press (0.3 < shoulder <= 0.9):")
  print(f"  Light press with L/R btn: {light_with_btn:,}")
  print(f"  Light press, no L/R btn:  {light_no_btn:,}")

  return {
      'btn_and_full': btn_and_full,
      'btn_no_full': btn_no_full,
      'full_no_btn': full_no_btn,
      'btn_equals_full': btn_no_full == 0 and full_no_btn == 0,
  }


def analyze_z_button_relationship(data: dict, shoulder_threshold: float = 0.3) -> dict:
  """Analyze if Z correlates with A and/or slight shoulder press.

  Z is a grab button. In Melee, pressing Z triggers a grab, which may
  correlate with light shield input.

  Args:
      data: Raw (unnormalized) controller data.
      shoulder_threshold: Threshold for detecting slight shoulder press.

  Returns:
      Dict with analysis results.
  """
  buttons = data['buttons']
  shoulder = data['shoulder']

  # Button order: A, B, X, Y, Z, L, R, D_UP
  A = buttons[:, 0].astype(bool)
  Z = buttons[:, 4].astype(bool)

  # Slight shoulder press (analog)
  slight_shoulder = shoulder > shoulder_threshold

  n_frames = len(buttons)
  n_z = Z.sum()

  # Check relationships
  z_with_a = (Z & A).sum()
  z_with_shoulder = (Z & slight_shoulder).sum()
  z_with_a_and_shoulder = (Z & A & slight_shoulder).sum()
  z_alone = (Z & ~A & ~slight_shoulder).sum()

  print("\n" + "="*60)
  print("Z BUTTON RELATIONSHIP ANALYSIS")
  print("="*60)

  print(f"\nZ button pressed: {n_z:,} frames ({n_z/n_frames*100:.3f}%)")
  if n_z > 0:
    print(
        f"  Z with A:              {z_with_a:,} ({z_with_a/n_z*100:.1f}% of Z presses)")
    print(
        f"  Z with shoulder>{shoulder_threshold}: {z_with_shoulder:,} ({z_with_shoulder/n_z*100:.1f}% of Z presses)")
    print(
        f"  Z with A AND shoulder: {z_with_a_and_shoulder:,} ({z_with_a_and_shoulder/n_z*100:.1f}% of Z presses)")
    print(
        f"  Z alone (no A/shoulder):{z_alone:,} ({z_alone/n_z*100:.1f}% of Z presses)")
  else:
    print("  (no Z presses in dataset)")

  z_implies_a_shoulder = n_z > 0 and z_alone == 0 and z_with_a_and_shoulder == n_z
  print(
      f"  -> Z implies A + shoulder>{shoulder_threshold}: {z_implies_a_shoulder}")

  return {
      'n_z': n_z,
      'z_with_a': z_with_a,
      'z_with_shoulder': z_with_shoulder,
      'z_with_a_and_shoulder': z_with_a_and_shoulder,
      'z_alone': z_alone,
      'z_implies_a_shoulder': z_implies_a_shoulder,
  }


def main():
  parser = argparse.ArgumentParser(
      description='Analyze button relationships in controller data.')
  parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR'),
                      help='Directory containing replay data files')
  parser.add_argument('--meta_path', default=os.environ.get('META_PATH'),
                      help='Path to metadata JSON file')
  parser.add_argument('--samples_per_character', type=int, default=50,
                      help='Number of replays to sample per character')
  parser.add_argument('--max_frames_per_game', type=int, default=1000,
                      help='Max frames to sample per game')
  parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for sampling')
  parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers for extraction')
  args = parser.parse_args()

  if not args.data_dir or not args.meta_path:
    raise ValueError("DATA_DIR and META_PATH must be set")

  print(f"Data dir: {args.data_dir}")
  print(f"Meta path: {args.meta_path}")

  # Sample replays
  sampled = sample_replays_by_character(
      args.meta_path,
      args.data_dir,
      samples_per_character=args.samples_per_character,
      seed=args.seed,
  )

  # Extract RAW controller data (no button normalization)
  print("\nExtracting raw controller data...")
  data = extract_controllers_from_replays(
      sampled,
      args.data_dir,
      max_frames_per_game=args.max_frames_per_game,
      normalize_buttons_flag=False,
      num_workers=args.num_workers,
  )

  print(f"Extracted {len(data['buttons']):,} frames of controller data")

  # Run analyses
  xy_results = analyze_button_equivalences(data)
  shoulder_results = analyze_shoulder_button_relationship(data)
  z_results = analyze_z_button_relationship(data)

  # Summary
  print("\n" + "="*60)
  print("SUMMARY")
  print("="*60)

  print("\n1. X and Y buttons:")
  if xy_results['x_equals_y']:
    print("   -> X=Y always holds (neither used, or perfectly correlated)")
  else:
    print("   -> X and Y are used independently")
    print("   -> Merging X→Y is a DESIGN CHOICE, not a data property")

  print("\n2. L and R buttons:")
  if xy_results['l_equals_r']:
    print("   -> L=R always holds (neither used, or perfectly correlated)")
  else:
    print("   -> L and R are used independently")
    print("   -> Merging L→R is a DESIGN CHOICE, not a data property")

  print("\n3. L/R button vs full shoulder press:")
  if shoulder_results['btn_equals_full']:
    print("   -> L/R button = full shoulder press (perfectly correlated)")
    print("   -> L/R button is REDUNDANT with analog shoulder value")
  else:
    print("   -> L/R button and full shoulder are NOT perfectly correlated")

  print("\n4. Z button relationship:")
  if z_results['n_z'] == 0:
    print("   -> Z button not used in dataset")
  elif z_results['z_implies_a_shoulder']:
    print("   -> Z implies A + slight shoulder press")
  else:
    a_pct = z_results['z_with_a'] / z_results['n_z'] * \
        100 if z_results['n_z'] > 0 else 0
    sh_pct = z_results['z_with_shoulder'] / \
        z_results['n_z'] * 100 if z_results['n_z'] > 0 else 0
    print(f"   -> Z with A: {a_pct:.1f}%, Z with shoulder: {sh_pct:.1f}%")
    if z_results['z_alone'] > 0:
      print(f"   -> Z can be pressed alone ({z_results['z_alone']} times)")

  return {
      'xy_results': xy_results,
      'shoulder_results': shoulder_results,
      'z_results': z_results,
  }


if __name__ == '__main__':
  main()
