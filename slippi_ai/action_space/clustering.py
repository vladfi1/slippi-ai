"""Analyze and cluster controller inputs from Slippi replays.

First we look at buttons and find the most common combinations.

"""

import collections
from dataclasses import dataclass
import json
import os
import itertools
import random
from typing import Optional
import typing as tp

import numpy as np
import melee
from melee import Character as C

from slippi_ai.data import read_table


@dataclass
class PolarBucketInfo:
  """Info returned by bucket_sticks_polar."""
  n_log_radius: int
  n_angle: int
  n_angle_per_radius: list[int]
  n_regular_classes: int
  total_classes: int
  at_origin_count: int
  regular_count: int
  min_radius: float
  max_radius: float


@dataclass
class DeltaBucketInfo:
  """Info returned by bucket_deltas_polar."""
  n_log_radius: int
  n_angle: int
  n_angle_per_radius: list[int]
  n_regular_classes: int
  total_classes: int
  no_movement_count: int
  dst_origin_count: int
  regular_count: int
  min_radius: float
  max_radius: float


@dataclass
class ButtonDataInfo:
  """Info returned by get_button_data."""
  button_names: list[str]
  n_binary_buttons: int
  light_shoulder_idx: int
  c_stick_idx: int
  light_press_freq: float
  c_stick_at_origin: int
  c_stick_buckets_used: int
  c_stick_total_buckets: int


@dataclass
class ComboClusterResult:
  """Result for a single button combo in hierarchical clustering."""
  combo: tuple
  count: int
  n_stick_clusters: int
  scaled_exclude_pct: float
  buckets_per_radius: list[int]


@dataclass
class HierarchicalClusterResult:
  """Result returned by cluster_hierarchical."""
  exclude_pct: float
  n_log_radius: int
  n_angle: int
  n_angle_per_radius: list[int]
  n_button_combos: int
  button_coverage: float
  total_clusters: int
  combo_results: list[ComboClusterResult]
  button_info: ButtonDataInfo


ALLOWED_CHARACTERS = {
    C.MARIO, C.FOX, C.CPTFALCON, C.DK, C.KIRBY, C.BOWSER, C.LINK, C.SHEIK,
    C.PEACH, C.POPO, C.NESS, C.SAMUS, C.YOSHI, C.PIKACHU, C.JIGGLYPUFF,
    C.MEWTWO, C.LUIGI, C.MARTH, C.ZELDA, C.YLINK, C.DOC, C.FALCO, C.PICHU,
    C.GAMEANDWATCH, C.GANONDORF, C.ROY,
}
assert len(ALLOWED_CHARACTERS) == 26


def sample_replays_by_character(
    meta_path: str,
    data_dir: str,
    allowed_characters: set[melee.Character] = ALLOWED_CHARACTERS,
    samples_per_character: int = 100,
    seed: int = 42,
) -> list:
  """Sample replays to get balanced representation across characters."""
  allowed_character_values = {char.value for char in allowed_characters}

  with open(meta_path) as f:
    meta_rows = json.load(f)

  # Group by character (p0 character)
  by_character = collections.defaultdict(list)
  for row in meta_rows:
    char = row['players'][0]['character']
    if char not in allowed_character_values:
      continue
    by_character[char].append(row)

  rng = random.Random(seed)
  sampled = []

  for char, rows in by_character.items():
    char_name = melee.Character(char).name
    n = min(samples_per_character, len(rows))
    sample = rng.sample(rows, n)
    sampled.extend(sample)
    print(f"{char_name}: {len(rows)} total, sampled {n}")

  print(f"\nTotal sampled: {len(sampled)} replays")
  return sampled


def normalize_buttons(buttons: np.ndarray) -> np.ndarray:
  """Normalize buttons by merging equivalent inputs.

  Merges:
    - X → Y (both are jump)
    - L → R (both are shield)

  Button order: A, B, X, Y, Z, L, R, D_UP
  """
  buttons = buttons.copy()
  # Merge X (index 2) into Y (index 3)
  buttons[:, 3] = buttons[:, 2] | buttons[:, 3]
  buttons[:, 2] = 0
  # Merge L (index 5) into R (index 6)
  buttons[:, 6] = buttons[:, 5] | buttons[:, 6]
  buttons[:, 5] = 0
  return buttons


def bucket_shoulder(shoulder: np.ndarray) -> np.ndarray:
  """Bucket shoulder values into 3 categories.

  Args:
      shoulder: Array of shoulder values in [0, 1].

  Returns:
      Array of bucket indices:
      - 0: unpressed (shoulder <= 0.3)
      - 1: light press (0.3 < shoulder <= 0.9)
      - 2: full press (shoulder > 0.9, correlates with R button)
  """
  labels = np.zeros(len(shoulder), dtype=np.uint8)
  labels[shoulder > 0.3] = 1
  labels[shoulder > 0.9] = 2
  return labels


def cluster_sticks_grid(
    stick: np.ndarray,
    resolution: int = 16,
    cutoffs: tp.Sequence[float] = (
        10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01),
) -> list[int]:
  """Discretize stick positions to a grid.

  Args:
      stick: Array of shape (n, 2) with stick positions in [0, 1].
      resolution: Grid resolution, must be a factor of 160.
      cutoffs: List of cutoff percentages for cluster size analysis.

  Returns:
      Number of clusters needed to cover (100 - cutoff)% of data at each cutoff.
  """
  assert 160 % resolution == 0, f"resolution must be a factor of 160, got {resolution}"

  # Discretize: map [0, 1] -> [0, resolution] via rounding
  grid_coords = np.rint(stick * resolution).astype(np.int32)
  grid_coords = np.clip(grid_coords, 0, resolution)

  counts = collections.Counter(map(tuple, grid_coords))

  total_points = len(stick)
  # Sort bucket sizes largest to smallest (greedy: pick biggest buckets first)
  bucket_sizes = sorted(counts.values(), reverse=True)
  bucket_index = 0
  cumulative = 0

  cutoff_indices = []
  for cutoff in cutoffs:
    # How many data points we need to cover
    points_needed = (1 - cutoff / 100) * total_points

    while cumulative < points_needed and bucket_index < len(bucket_sizes):
      cumulative += bucket_sizes[bucket_index]
      bucket_index += 1

    cutoff_indices.append(bucket_index)

  return cutoff_indices


def _extract_single_replay(
    row: dict,
    data_dir: str,
    max_frames_per_game: Optional[int],
    normalize_buttons_flag: bool,
    seed: Optional[int] = None,
) -> Optional[dict]:
  """Extract controller data from a single replay.

  Args:
      row: Metadata dict for the replay.
      data_dir: Directory containing replay data files.
      max_frames_per_game: Max frames to sample per game (None = all).
      normalize_buttons_flag: If True, merge equivalent buttons (X→Y, L→R).
      seed: Random seed for frame sampling (for reproducibility in multiprocessing).

  Returns:
      Dict with 'buttons', 'main_stick', 'c_stick', 'shoulder' arrays,
      or None if the replay could not be read.
  """
  path = os.path.join(data_dir, row['slp_md5'])
  try:
    game = read_table(path, compressed=True)
  except Exception:
    return None

  # Use seed if provided (for reproducibility in multiprocessing)
  rng = np.random.RandomState(seed)

  all_buttons = []
  all_main_sticks = []
  all_c_sticks = []
  all_shoulders = []
  all_main_stick_deltas = []
  all_c_stick_deltas = []
  all_main_stick_delta_dsts = []
  all_c_stick_delta_dsts = []

  # Extract controller for both players
  for player in [game.p0, game.p1]:
    ctrl = player.controller

    # Build full stick arrays first (for computing deltas on consecutive frames)
    full_main_stick = np.stack([
        ctrl.main_stick.x,
        ctrl.main_stick.y,
    ], axis=1).astype(np.float32)

    full_c_stick = np.stack([
        ctrl.c_stick.x,
        ctrl.c_stick.y,
    ], axis=1).astype(np.float32)

    # Compute deltas on consecutive frames before any sampling
    # Delta[i] = stick[i+1] - stick[i], so destination is stick[i+1]
    main_stick_deltas = np.diff(full_main_stick, axis=0)
    c_stick_deltas = np.diff(full_c_stick, axis=0)
    main_stick_delta_dsts = full_main_stick[1:]  # destinations
    c_stick_delta_dsts = full_c_stick[1:]

    # Limit frames if requested
    n_frames = len(ctrl.buttons.A)
    if max_frames_per_game and n_frames > max_frames_per_game:
      indices = rng.choice(n_frames, max_frames_per_game, replace=False)
      # For deltas, sample from n_frames-1 (since deltas have one fewer element)
      delta_indices = rng.choice(
          n_frames - 1, max_frames_per_game, replace=False)
    else:
      indices = np.arange(n_frames)
      delta_indices = np.arange(n_frames - 1)

    # Buttons as a single array (8 buttons)
    buttons = np.stack([
        ctrl.buttons.A[indices],
        ctrl.buttons.B[indices],
        ctrl.buttons.X[indices],
        ctrl.buttons.Y[indices],
        ctrl.buttons.Z[indices],
        ctrl.buttons.L[indices],
        ctrl.buttons.R[indices],
        ctrl.buttons.D_UP[indices],
    ], axis=1).astype(np.uint8)

    main_stick = full_main_stick[indices]
    c_stick = full_c_stick[indices]
    shoulder = ctrl.shoulder[indices].astype(np.float32)

    # Apply button normalization if requested (X→Y, L→R merge)
    if normalize_buttons_flag:
      buttons = normalize_buttons(buttons)

    all_buttons.append(buttons)
    all_main_sticks.append(main_stick)
    all_c_sticks.append(c_stick)
    all_shoulders.append(shoulder)
    all_main_stick_deltas.append(main_stick_deltas[delta_indices])
    all_c_stick_deltas.append(c_stick_deltas[delta_indices])
    all_main_stick_delta_dsts.append(main_stick_delta_dsts[delta_indices])
    all_c_stick_delta_dsts.append(c_stick_delta_dsts[delta_indices])

  return {
      'buttons': np.concatenate(all_buttons, axis=0),
      'main_stick': np.concatenate(all_main_sticks, axis=0),
      'c_stick': np.concatenate(all_c_sticks, axis=0),
      'shoulder': np.concatenate(all_shoulders, axis=0),
      'main_stick_deltas': np.concatenate(all_main_stick_deltas, axis=0),
      'c_stick_deltas': np.concatenate(all_c_stick_deltas, axis=0),
      'main_stick_delta_dsts': np.concatenate(all_main_stick_delta_dsts, axis=0),
      'c_stick_delta_dsts': np.concatenate(all_c_stick_delta_dsts, axis=0),
  }


def _extract_worker(args: tuple) -> Optional[dict]:
  """Worker function for multiprocessing."""
  return _extract_single_replay(*args)


def extract_controllers_from_replays(
    sampled_meta: list,
    data_dir: str,
    max_frames_per_game: Optional[int] = None,
    normalize_buttons_flag: bool = True,
    num_workers: int = 1,
    show_progress: bool = True,
) -> dict:
  """Extract controller data from sampled replays.

  Args:
      sampled_meta: List of replay metadata dicts.
      data_dir: Directory containing replay data files.
      max_frames_per_game: Max frames to sample per game (None = all).
      normalize_buttons_flag: If True, merge equivalent buttons (X→Y, L→R).
      num_workers: Number of parallel workers (1 = single-threaded).
      show_progress: If True, show progress bar with tqdm.

  Returns:
      Dict with 'buttons', 'main_stick', 'c_stick', 'shoulder' arrays.
      Shoulder and sticks are kept at native resolution for later clustering.
  """
  import concurrent.futures
  try:
    import tqdm
    has_tqdm = True
  except ImportError:
    has_tqdm = False

  # Prepare arguments for each replay
  args_list = [
      (row, data_dir, max_frames_per_game, normalize_buttons_flag, i)
      for i, row in enumerate(sampled_meta)
  ]

  results = []

  # Create progress iterator
  def make_progress_iter(iterable, total):
    if show_progress and has_tqdm:
      return tqdm.tqdm(iterable, total=total, desc="Extracting", unit="replay")
    elif show_progress:
      # Simple progress without tqdm
      processed = [0]

      def progress_iter():
        for item in iterable:
          yield item
          processed[0] += 1
          if processed[0] % 100 == 0:
            print(f"Processing replay {processed[0]}/{total}...")
      return progress_iter()
    else:
      return iterable

  if num_workers == 1:
    # Single-threaded processing
    for args in make_progress_iter(args_list, len(args_list)):
      result = _extract_single_replay(*args)
      if result is not None:
        results.append(result)
  else:
    # Multi-process processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
      try:
        futures = [executor.submit(_extract_worker, args)
                   for args in args_list]

        for future in make_progress_iter(
            concurrent.futures.as_completed(futures),
            len(futures)
        ):
          result = future.result()
          if result is not None:
            results.append(result)
      except KeyboardInterrupt:
        print('KeyboardInterrupt, shutting down')
        executor.shutdown(cancel_futures=True)
        raise

  if not results:
    raise ValueError("No replays could be read")

  # Concatenate all results
  return {
      'buttons': np.concatenate([r['buttons'] for r in results], axis=0),
      'main_stick': np.concatenate([r['main_stick'] for r in results], axis=0),
      'c_stick': np.concatenate([r['c_stick'] for r in results], axis=0),
      'shoulder': np.concatenate([r['shoulder'] for r in results], axis=0),
      'main_stick_deltas': np.concatenate([r['main_stick_deltas'] for r in results], axis=0),
      'c_stick_deltas': np.concatenate([r['c_stick_deltas'] for r in results], axis=0),
      'main_stick_delta_dsts': np.concatenate([r['main_stick_delta_dsts'] for r in results], axis=0),
      'c_stick_delta_dsts': np.concatenate([r['c_stick_delta_dsts'] for r in results], axis=0),
  }


def analyze_buttons(
    data: dict,
    c_stick_n_log_radius: int = 3,
    c_stick_n_angle: int = 8,
) -> dict:
  """Analyze button combinations and their frequencies.

  Args:
      data: Dict with 'buttons', 'shoulder', 'c_stick'.
      c_stick_n_log_radius: Number of log-radius buckets for c-stick (default 3).
      c_stick_n_angle: Number of angle buckets for c-stick (default 8).

  Returns:
      Dict with 'counter' and 'button_names'.
  """
  print("\n" + "="*60)
  print("BUTTON ANALYSIS")
  print("="*60)

  buttons = data['buttons']
  button_names = ['A', 'B', 'X', 'Y', 'Z', 'L', 'R', 'D_UP']

  # Individual button frequencies
  print("\nIndividual button press frequencies:")
  for i, name in enumerate(button_names):
    freq = buttons[:, i].mean()
    print(f"  {name}: {freq:.4f} ({freq*100:.2f}%)")

  # Get combined button data (includes light_shoulder and c_stick)
  combined_buttons, button_counter, info = get_button_data(
      data, c_stick_n_log_radius=c_stick_n_log_radius, c_stick_n_angle=c_stick_n_angle)

  # Print light shoulder stats
  print(
      f"\nLight shoulder press frequency: {info.light_press_freq:.4f} ({info.light_press_freq*100:.2f}%)")

  # Print c-stick stats
  print(
      f"\nC-stick polar buckets ({c_stick_n_log_radius}, {c_stick_n_angle}):")
  print(
      f"  At origin (bucket 0): {info.c_stick_at_origin:,} ({info.c_stick_at_origin/len(buttons)*100:.2f}%)")
  print(
      f"  Total buckets used: {info.c_stick_buckets_used} / {info.c_stick_total_buckets}")

  print(f"\nTotal frames: {len(buttons):,}")
  print(f"Unique button combinations: {len(button_counter)}")

  # Most common combinations
  # Don't print c-stick label if it's 0 (neutral)
  print("\nTop 30 button combinations:")
  for combo, count in button_counter.most_common(30):
    pct = count / len(buttons) * 100
    combo_str = format_button_combo(combo, info)
    print(f"  {combo_str:30s}: {count:>10,} ({pct:>6.2f}%)")

  # Coverage analysis
  print("\nCoverage by number of clusters:")
  sorted_counts = sorted(button_counter.values(), reverse=True)
  cumsum = np.cumsum(sorted_counts)
  total = len(buttons)
  for threshold in [0.90, 0.95, 0.99, 0.999]:
    n_needed = np.searchsorted(cumsum, threshold * total) + 1
    print(f"  {threshold*100:.1f}% coverage: {n_needed} combinations")

  return {
      'counter': button_counter,
      'button_names': info.button_names,
  }


def analyze_shoulder(shoulder: np.ndarray) -> dict:
  """Analyze shoulder/trigger values."""
  print("\n" + "="*60)
  print("SHOULDER/TRIGGER ANALYSIS")
  print("="*60)

  print(f"Min: {shoulder.min():.4f}, Max: {shoulder.max():.4f}")
  print(f"Mean: {shoulder.mean():.4f}, Std: {shoulder.std():.4f}")

  # Histogram of values
  print("\nValue distribution:")
  edges = [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01]
  hist, _ = np.histogram(shoulder, bins=edges)
  for i in range(len(hist)):
    pct = hist[i] / len(shoulder) * 100
    print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f}): {hist[i]:>12,} ({pct:>6.2f}%)")

  # Unique values (rounded)
  rounded = np.round(shoulder, 2)
  unique_vals, counts = np.unique(rounded, return_counts=True)
  print(f"\nUnique values (rounded to 0.01): {len(unique_vals)}")

  # Most common
  sorted_idx = np.argsort(-counts)[:20]
  print("\nTop 20 shoulder values:")
  for idx in sorted_idx:
    pct = counts[idx] / len(shoulder) * 100
    print(f"  {unique_vals[idx]:.2f}: {counts[idx]:>12,} ({pct:>6.2f}%)")

  return {'unique_rounded': unique_vals, 'counts': counts}


def analyze_stick(stick: np.ndarray, name: str) -> dict:
  """Analyze stick positions."""
  print("\n" + "="*60)
  print(f"{name.upper()} STICK ANALYSIS")
  print("="*60)

  x, y = stick[:, 0], stick[:, 1]

  print(f"X - Min: {x.min():.4f}, Max: {x.max():.4f}, Mean: {x.mean():.4f}")
  print(f"Y - Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}")

  # Round to see unique positions
  raw_sticks = np.round(stick * 160)
  unique_positions = set(map(tuple, raw_sticks))
  print(f"\nUnique positions (rounded to 0.01): {len(unique_positions)}")

  # Count positions
  pos_tuples = [tuple(row) for row in raw_sticks]
  counter = collections.Counter(pos_tuples)

  print("\nTop 30 stick positions:")
  for pos, count in counter.most_common(30):
    pct = count / len(stick) * 100
    # Convert to centered coordinates for display
    cx, cy = (pos[0] - 80) / 80, (pos[1] - 80) / 80
    print(f"  ({cx:>+6.2f}, {cy:>+6.2f}): {count:>12,} ({pct:>6.2f}%)")

  # Coverage analysis using grid clustering
  cutoffs = [10, 5, 2, 1, 0.5, 0.2, 0.1]
  print("\nClusters needed by resolution and cutoff:")
  print(f"  {'Resolution':>10} | " + " ".join(f"{c:>6}%" for c in cutoffs))
  print("  " + "-" * (13 + 8 * len(cutoffs)))
  for resolution in [16, 32, 40, 80, 160]:
    n_clusters = cluster_sticks_grid(
        stick, resolution=resolution, cutoffs=cutoffs)
    print(f"  {resolution:>10} | " + " ".join(f"{n:>7}" for n in n_clusters))

  # Analyze by region (centered coordinates: 0.5 is neutral)
  # Convert to centered: x-0.5, y-0.5 so neutral is (0,0)
  cx, cy = x - 0.5, y - 0.5
  threshold = 0.2  # deadzone threshold

  print("\nRegion analysis (centered coords, threshold=0.2):")
  neutral = (np.abs(cx) < threshold) & (np.abs(cy) < threshold)
  print(f"  Neutral: {neutral.sum():,} ({neutral.mean()*100:.2f}%)")

  # Cardinal directions
  for direction, cond in [
      ("Right", (cx > threshold) & (np.abs(cy) < threshold)),
      ("Left", (cx < -threshold) & (np.abs(cy) < threshold)),
      ("Up", (cy > threshold) & (np.abs(cx) < threshold)),
      ("Down", (cy < -threshold) & (np.abs(cx) < threshold)),
  ]:
    print(f"  {direction}: {cond.sum():,} ({cond.mean()*100:.2f}%)")

  # Diagonals
  for direction, cond in [
      ("Up-Right", (cx > threshold) & (cy > threshold)),
      ("Up-Left", (cx < -threshold) & (cy > threshold)),
      ("Down-Right", (cx > threshold) & (cy < -threshold)),
      ("Down-Left", (cx < -threshold) & (cy < -threshold)),
  ]:
    print(f"  {direction}: {cond.sum():,} ({cond.mean()*100:.2f}%)")

  return {'counter': counter, 'unique_positions': len(unique_positions), 'data': stick}


def bucket_sticks_polar(
    stick: np.ndarray,
    n_log_radius: int = 4,
    n_angle: int = 8,
    origin_threshold: float = 0.05,
) -> tuple[np.ndarray, PolarBucketInfo]:
  """Bucket stick positions using polar coordinates with log-radius and angle.

  Positions are relative to center (0.5, 0.5).

  Angle granularity scales with radius based on actual data: at the outermost
  radius we use n_angle buckets, at innermost we use n_angle / radius_ratio
  buckets. For bucket k, the effective radius is min_r * ratio^(k/(n-1)),
  so n_angle_k = n_angle * ratio^(k/(n-1) - 1).

  Special class:
  - 0: at origin (within threshold)

  Regular classes start at 1, computed as:
      sum(angle_buckets for previous radius levels) + angle_bucket + 1

  Args:
      stick: Array of shape (n, 2) with stick positions in [0, 1].
      n_log_radius: Number of log-radius buckets.
      n_angle: Max number of angle buckets (at outermost radius).
      origin_threshold: Threshold for considering position as origin.

  Returns:
      (labels, info) where labels are bucket indices and info contains bucket details.
  """
  n = len(stick)
  labels = np.zeros(n, dtype=np.int32)

  # Compute position relative to center
  cx = stick[:, 0] - 0.5
  cy = stick[:, 1] - 0.5
  radius = np.sqrt(cx**2 + cy**2)
  angle = np.arctan2(cy, cx)  # [-pi, pi]

  # Class 0: at origin
  at_origin = radius < origin_threshold
  labels[at_origin] = 0

  # Regular classes for non-origin positions
  regular = ~at_origin

  # Default values if no regular points
  n_angle_per_radius = [n_angle] * n_log_radius
  min_radius = max_radius = 0.0

  if regular.any():
    reg_radius = radius[regular]
    reg_angle = angle[regular]

    # Log-radius bucketing
    min_radius = np.min(reg_radius)
    max_radius = np.max(reg_radius)
    ratio = max_radius / min_radius

    # Compute angle buckets per radius level based on actual radius ratio
    # At bucket k, effective radius is min_r * ratio^(k/(n-1))
    # n_angle_k = n_angle * (r_k / max_r) = n_angle * ratio^(k/(n-1) - 1)
    # Round down to nearest power of 2
    def floor_pow2(x):
      if x < 1:
        return 1
      return 2 ** int(np.floor(np.log2(x)))

    if n_log_radius > 1:
      n_angle_per_radius = [
          floor_pow2(n_angle * ratio ** (k / (n_log_radius - 1) - 1))
          for k in range(n_log_radius)
      ]
    else:
      n_angle_per_radius = [n_angle]

    # Cumulative offset for each radius level (for label computation)
    radius_label_offset = np.cumsum([0] + n_angle_per_radius[:-1])

    log_r = np.log(reg_radius)
    scaled_log_r = (log_r - np.log(min_radius)) / \
        (np.log(ratio)) if ratio > 1 else np.zeros_like(log_r)

    log_bucket = np.rint(scaled_log_r * (n_log_radius - 1)).astype(np.int32)

    # Angle bucketing with radius-dependent granularity
    angle_normalized = (reg_angle + np.pi) / (2 * np.pi)  # [0, 1]

    # Compute angle bucket for each point based on its radius bucket
    reg_labels = np.zeros(len(reg_radius), dtype=np.int32)
    for r in range(n_log_radius):
      mask = log_bucket == r
      if mask.any():
        n_ang = n_angle_per_radius[r]
        ang_bucket = np.rint(angle_normalized[mask] * n_ang).astype(np.int32)
        # Wrap around (angle n_ang == angle 0)
        ang_bucket[ang_bucket == n_ang] = 0
        # Label = offset for this radius + angle bucket + 1 (for origin class)
        reg_labels[mask] = radius_label_offset[r] + ang_bucket + 1

    labels[regular] = reg_labels

  n_regular_classes = sum(n_angle_per_radius)
  total_classes = 1 + n_regular_classes

  info = PolarBucketInfo(
      n_log_radius=n_log_radius,
      n_angle=n_angle,
      n_angle_per_radius=n_angle_per_radius,
      n_regular_classes=n_regular_classes,
      total_classes=total_classes,
      at_origin_count=int(at_origin.sum()),
      regular_count=int(regular.sum()),
      min_radius=float(min_radius),
      max_radius=float(max_radius),
  )

  return labels, info


def analyze_stick_polar(
    stick: np.ndarray,
    name: str,
    n_log_radius_values: list[int] = [2, 3, 4, 5, 6],
    n_angle_values: list[int] = [4, 8, 16, 32, 64],
) -> dict:
  """Analyze stick positions using polar bucketing.

  Args:
      stick: Array of shape (n, 2) with stick positions in [0, 1].
      name: Name for display (e.g., 'main', 'c').
      n_log_radius_values: List of log-radius bucket counts to try.
      n_angle_values: List of angle bucket counts to try.
  """
  print("\n" + "="*60)
  print(f"{name.upper()} STICK ANALYSIS (Polar)")
  print("="*60)

  n_total = len(stick)

  # Position relative to center
  raw_stick = np.rint((stick - 0.5) * 160)
  cx, cy = raw_stick.T
  radius = np.sqrt(cx**2 + cy**2)

  print(
      f"Radius from center - Mean: {radius.mean():.4f}, Std: {radius.std():.4f}, Max: {radius.max():.4f}")

  min_non_zero_radius = radius[radius > 0].min()
  print(f"Min non-zero radius: {min_non_zero_radius:.4f}")

  # Origin stats
  at_origin = radius < 0.05
  print(
      f"\nAt origin (radius < 0.05): {at_origin.sum():,} ({at_origin.mean()*100:.2f}%)")
  print(
      f"Regular positions:         {(~at_origin).sum():,} ({(~at_origin).mean()*100:.2f}%)")

  # Coverage analysis with different bucket configurations
  cutoffs = [10, 5, 2, 1, 0.5, 0.2, 0.1]
  max_log_r = max(n_log_radius_values)
  radius_headers = " ".join(f"r{i}" for i in range(max_log_r))
  print(f"\nBuckets needed by (log_radius, angle) and cutoff:")
  print(f"  {'(logR, angle)':>15} {'classes':>8} | " +
        " ".join(f"{c:>6}%" for c in cutoffs) + f" | {radius_headers}")
  print("  " + "-" * (27 + 8 * len(cutoffs) + 3 + 3 * max_log_r))

  for n_log_r in n_log_radius_values:
    for n_ang in n_angle_values:
      labels, info = bucket_sticks_polar(
          stick, n_log_radius=n_log_r, n_angle=n_ang)

      # Count buckets and compute coverage
      counter = collections.Counter(labels)
      total_classes = info.total_classes
      bucket_sizes = sorted(counter.values(), reverse=True)
      cumsum = np.cumsum(bucket_sizes)

      n_buckets = []
      for cutoff in cutoffs:
        points_needed = (1 - cutoff / 100) * n_total
        idx = np.searchsorted(cumsum, points_needed) + 1
        n_buckets.append(min(idx, len(bucket_sizes)))

      config_str = f"({n_log_r}, {n_ang})"
      # Pad angle counts to max_log_r columns
      angle_counts = info.n_angle_per_radius + [''] * (max_log_r - n_log_r)
      angles_str = " ".join(
          f"{a:>2}" if a != '' else "  " for a in angle_counts)
      print(f"  {config_str:>15} {total_classes:>8} | " +
            " ".join(f"{n:>7}" for n in n_buckets) + f" | {angles_str}")

  return {'stick': stick}


def bucket_deltas_polar(
    deltas: np.ndarray,
    destinations: np.ndarray,
    n_log_radius: int = 4,
    n_angle: int = 8,
    origin_threshold: float = 0.05,
) -> tuple[np.ndarray, DeltaBucketInfo]:
  """Bucket deltas using polar coordinates with log-radius and angle.

  Angle granularity scales with radius based on actual data: at the outermost
  radius we use n_angle buckets, at innermost we use n_angle / radius_ratio
  buckets. For bucket k, the effective radius is min_r * ratio^(k/(n-1)),
  so n_angle_k = n_angle * ratio^(k/(n-1) - 1).

  Special classes:
  - 0: no movement (radius=0)
  - 1: destination is origin

  Regular classes start at 2, computed as:
      sum(angle_buckets for previous radius levels) + angle_bucket + 2

  Args:
      deltas: Array of shape (n, 2) with delta values.
      destinations: Array of shape (n, 2) with destination positions.
      n_log_radius: Number of log-radius buckets.
      n_angle: Max number of angle buckets (at outermost radius).
      origin_threshold: Threshold for considering destination as origin.

  Returns:
      (labels, info) where labels are bucket indices and info contains bucket details.
  """
  n = len(deltas)
  labels = np.zeros(n, dtype=np.int32)

  # Compute radius and angle
  dx, dy = deltas[:, 0], deltas[:, 1]
  radius = np.sqrt(dx**2 + dy**2)
  angle = np.arctan2(dy, dx)  # [-pi, pi]

  # Destination distance from origin (origin is at 0.5, 0.5)
  dst_dx = destinations[:, 0] - 0.5
  dst_dy = destinations[:, 1] - 0.5
  dst_radius = np.sqrt(dst_dx**2 + dst_dy**2)

  # Class 0: no movement
  no_movement = radius == 0
  labels[no_movement] = 0

  # Class 1: destination is origin
  dst_is_origin = (dst_radius < origin_threshold) & ~no_movement
  labels[dst_is_origin] = 1

  # Regular classes for moving deltas not going to origin
  regular = ~no_movement & ~dst_is_origin

  # Default values if no regular points
  n_angle_per_radius = [n_angle] * n_log_radius
  min_radius = max_radius = 0.0

  if regular.any():
    reg_radius = radius[regular]
    reg_angle = angle[regular]

    # Log-radius bucketing
    min_radius = np.min(reg_radius)
    assert min_radius > 0, "All regular radii should be > 0"
    max_radius = np.max(reg_radius)
    ratio = max_radius / min_radius

    # Compute angle buckets per radius level based on actual radius ratio
    # At bucket k, effective radius is min_r * ratio^(k/(n-1))
    # n_angle_k = n_angle * (r_k / max_r) = n_angle * ratio^(k/(n-1) - 1)
    # Round down to nearest power of 2
    def floor_pow2(x):
      if x < 1:
        return 1
      return 2 ** int(np.floor(np.log2(x)))

    if n_log_radius > 1:
      n_angle_per_radius = [
          floor_pow2(n_angle * ratio ** (k / (n_log_radius - 1) - 1))
          for k in range(n_log_radius)
      ]
    else:
      n_angle_per_radius = [n_angle]

    # Cumulative offset for each radius level (for label computation)
    radius_label_offset = np.cumsum([0] + n_angle_per_radius[:-1])

    log_r = np.log(reg_radius)
    scaled_log_r = (log_r - np.log(min_radius)) / \
        (np.log(ratio)) if ratio > 1 else np.zeros_like(log_r)

    log_bucket = np.rint(scaled_log_r * (n_log_radius - 1)).astype(np.int32)

    # Angle bucketing with radius-dependent granularity
    angle_normalized = (reg_angle + np.pi) / (2 * np.pi)  # [0, 1]

    # Compute angle bucket for each point based on its radius bucket
    reg_labels = np.zeros(len(reg_radius), dtype=np.int32)
    for r in range(n_log_radius):
      mask = log_bucket == r
      if mask.any():
        n_ang = n_angle_per_radius[r]
        ang_bucket = np.rint(angle_normalized[mask] * n_ang).astype(np.int32)
        # Wrap around (angle n_ang == angle 0)
        ang_bucket[ang_bucket == n_ang] = 0
        # Label = offset for this radius + angle bucket + 2 (for special classes)
        reg_labels[mask] = radius_label_offset[r] + ang_bucket + 2

    labels[regular] = reg_labels

  n_regular_classes = sum(n_angle_per_radius)
  total_classes = 2 + n_regular_classes

  info = DeltaBucketInfo(
      n_log_radius=n_log_radius,
      n_angle=n_angle,
      n_angle_per_radius=n_angle_per_radius,
      n_regular_classes=n_regular_classes,
      total_classes=total_classes,
      no_movement_count=int(no_movement.sum()),
      dst_origin_count=int(dst_is_origin.sum()),
      regular_count=int(regular.sum()),
      min_radius=float(min_radius),
      max_radius=float(max_radius),
  )

  return labels, info


def analyze_stick_deltas(
    deltas: np.ndarray,
    destinations: np.ndarray,
    name: str,
    n_log_radius_values: list[int] = [2, 3, 4, 5, 6],
    n_angle_values: list[int] = [4, 8, 16, 32],
) -> dict:
  """Analyze deltas between consecutive stick positions using polar bucketing.

  Args:
      deltas: Pre-computed deltas array of shape (n, 2) with values in [-1, 1].
      destinations: Destination positions array of shape (n, 2).
      name: Name for display (e.g., 'main', 'c').
      n_log_radius_values: List of log-radius bucket counts to try.
      n_angle_values: List of angle bucket counts to try.
  """
  print("\n" + "="*60)
  print(f"{name.upper()} STICK DELTA ANALYSIS (Polar)")
  print("="*60)

  dx, dy = deltas[:, 0], deltas[:, 1]
  n_total = len(deltas)

  print(
      f"dX - Min: {dx.min():.4f}, Max: {dx.max():.4f}, Mean: {dx.mean():.4f}, Std: {dx.std():.4f}")
  print(
      f"dY - Min: {dy.min():.4f}, Max: {dy.max():.4f}, Mean: {dy.mean():.4f}, Std: {dy.std():.4f}")

  # Magnitude of deltas
  radius = np.sqrt(dx**2 + dy**2)
  print(
      f"\nRadius - Mean: {radius.mean():.4f}, Std: {radius.std():.4f}, Max: {radius.max():.4f}")

  # Special class stats
  no_movement = (deltas == 0).all(axis=1)
  dst_dx = destinations[:, 0] - 0.5
  dst_dy = destinations[:, 1] - 0.5
  dst_radius = np.sqrt(dst_dx**2 + dst_dy**2)
  dst_is_origin = (dst_radius < 0.05) & ~no_movement

  print(f"\nSpecial classes:")
  print(
      f"  No movement (radius=0): {no_movement.sum():,} ({no_movement.mean()*100:.2f}%)")
  print(
      f"  Destination=origin:     {dst_is_origin.sum():,} ({dst_is_origin.mean()*100:.2f}%)")
  print(
      f"  Regular deltas:         {(~no_movement & ~dst_is_origin).sum():,} ({(~no_movement & ~dst_is_origin).mean()*100:.2f}%)")

  # Coverage analysis with different bucket configurations
  cutoffs = [10, 5, 2, 1, 0.5, 0.2, 0.1]
  max_log_r = max(n_log_radius_values)
  radius_headers = " ".join(f"r{i}" for i in range(max_log_r))
  print(f"\nBuckets needed by (log_radius, angle) and cutoff:")
  print(f"  {'(logR, angle)':>15} {'classes':>8} | " +
        " ".join(f"{c:>6}%" for c in cutoffs) + f" | {radius_headers}")
  print("  " + "-" * (27 + 8 * len(cutoffs) + 3 + 3 * max_log_r))

  for n_log_r in n_log_radius_values:
    for n_ang in n_angle_values:
      labels, info = bucket_deltas_polar(
          deltas, destinations, n_log_radius=n_log_r, n_angle=n_ang)

      # Count buckets and compute coverage
      counter = collections.Counter(labels)
      total_classes = info.total_classes
      bucket_sizes = sorted(counter.values(), reverse=True)
      cumsum = np.cumsum(bucket_sizes)

      n_buckets = []
      for cutoff in cutoffs:
        points_needed = (1 - cutoff / 100) * n_total
        idx = np.searchsorted(cumsum, points_needed) + 1
        n_buckets.append(min(idx, len(bucket_sizes)))

      config_str = f"({n_log_r}, {n_ang})"
      # Pad angle counts to max_log_r columns
      angle_counts = info.n_angle_per_radius + [''] * (max_log_r - n_log_r)
      angles_str = " ".join(
          f"{a:>2}" if a != '' else "  " for a in angle_counts)
      print(f"  {config_str:>15} {total_classes:>8} | " +
            " ".join(f"{n:>7}" for n in n_buckets) + f" | {angles_str}")

  return {'deltas': deltas, 'destinations': destinations}

  return {'counter': counter, 'unique_deltas': len(unique_deltas), 'data': deltas}


BUTTON_NAMES = ['A', 'B', 'X', 'Y', 'Z', 'L', 'R', 'D_UP']
N_BINARY_BUTTONS = len(BUTTON_NAMES)


def get_button_data(
    data: dict,
    c_stick_n_log_radius: int = 3,
    c_stick_n_angle: int = 8,
) -> tuple[np.ndarray, collections.Counter, ButtonDataInfo]:
  """Get combined button data including shoulder and c-stick.

  Args:
      data: Dict with 'buttons', 'shoulder', 'c_stick'.
      c_stick_n_log_radius: Number of log-radius buckets for c-stick.
      c_stick_n_angle: Number of angle buckets for c-stick.

  Returns:
      (combined_buttons, button_counter, info) where:
      - combined_buttons: array with buttons, light_shoulder, c_stick bucket
      - button_counter: Counter of button tuples
      - info: ButtonDataInfo with metadata for display
  """
  buttons = data['buttons']

  # Light shoulder press
  shoulder_labels = bucket_shoulder(data['shoulder'])
  light_press = (shoulder_labels == 1).astype(np.uint8)

  # C-stick polar bucket (re-ordered by frequency)
  c_stick_labels_raw, c_stick_polar_info = bucket_sticks_polar(
      data['c_stick'], n_log_radius=c_stick_n_log_radius, n_angle=c_stick_n_angle)

  # Re-order c-stick labels by frequency (0 stays as 0, others sorted by frequency)
  label_counts = collections.Counter(c_stick_labels_raw)
  non_zero_labels = [(label, count)
                     for label, count in label_counts.items() if label != 0]
  non_zero_labels.sort(key=lambda x: -x[1])
  label_map = {0: 0}
  for new_label, (old_label, _) in enumerate(non_zero_labels, start=1):
    label_map[old_label] = new_label
  c_stick_labels = np.array([label_map[l]
                            for l in c_stick_labels_raw], dtype=np.int32)

  combined_buttons = np.concatenate([
      buttons,
      light_press[:, None],
      c_stick_labels[:, None],
  ], axis=1)
  button_tuples = [tuple(row) for row in combined_buttons]

  # Count button combinations
  button_counter = collections.Counter(button_tuples)

  # Info for display
  info = ButtonDataInfo(
      button_names=BUTTON_NAMES + ['light_shoulder', 'c_stick'],
      n_binary_buttons=N_BINARY_BUTTONS,
      light_shoulder_idx=N_BINARY_BUTTONS,
      c_stick_idx=N_BINARY_BUTTONS + 1,
      light_press_freq=float(light_press.mean()),
      c_stick_at_origin=int((c_stick_labels == 0).sum()),
      c_stick_buckets_used=len(label_counts),
      c_stick_total_buckets=c_stick_polar_info.total_classes,
  )

  return combined_buttons, button_counter, info


def format_button_combo(combo: tuple, info: ButtonDataInfo) -> str:
  """Format a button combination tuple as a human-readable string.

  Args:
      combo: Tuple of button values.
      info: ButtonDataInfo from get_button_data.

  Returns:
      String like "A+B+light_shoulder+c3" or "(none)".
  """
  parts = []
  n_binary = info.n_binary_buttons
  c_stick_idx = info.c_stick_idx

  for i, v in enumerate(combo):
    if i < n_binary:
      # Binary buttons: include if pressed
      if v:
        parts.append(BUTTON_NAMES[i])
    elif i == c_stick_idx:
      # C-stick: include if not neutral (bucket > 0)
      if v > 0:
        parts.append(f'c{v}')
    else:
      # light_shoulder: include if pressed
      if v:
        parts.append('light_sh')

  return '+'.join(parts) if parts else '(none)'


def cluster_hierarchical(
    data: dict,
    exclude_pct: float = 0.01,
    n_log_radius: int = 4,
    n_angle: int = 8,
    button_data: Optional[tuple[np.ndarray,
                                collections.Counter, ButtonDataInfo]] = None,
) -> HierarchicalClusterResult:
  """Cluster actions hierarchically: first buttons, then main stick per button combo.

  Args:
      data: Dict with 'buttons', 'shoulder', 'main_stick', 'c_stick'.
      exclude_pct: Fraction of data to exclude (e.g., 0.01 = keep 99%).
      n_log_radius: Number of log-radius buckets for polar stick clustering.
      n_angle: Number of angle buckets for polar stick clustering.
      button_data: Optional pre-computed button data from get_button_data.

  Returns:
      HierarchicalClusterResult with clustering results.
  """
  buttons = data['buttons']
  main_stick = data['main_stick']
  n_frames = len(buttons)

  if button_data is None:
    combined_buttons, button_counter, info = get_button_data(data)
  else:
    combined_buttons, button_counter, info = button_data

  # Get representative n_angle_per_radius from full main_stick data
  _, polar_info = bucket_sticks_polar(
      main_stick, n_log_radius=n_log_radius, n_angle=n_angle)
  n_angle_per_radius = polar_info.n_angle_per_radius

  sorted_combos = button_counter.most_common()

  # Find button combinations needed for coverage
  cumsum = 0
  target = n_frames * (1 - exclude_pct)
  kept_combos = []
  for combo, count in sorted_combos:
    kept_combos.append((combo, count))
    cumsum += count
    if cumsum >= target:
      break

  n_button_combos = len(kept_combos)
  button_coverage = cumsum / n_frames

  # For each button combination, cluster main stick using polar buckets
  # Scale exclude_pct by sqrt(total/group_size) for smaller groups
  total_clusters = 0
  combo_results = []

  for combo, combo_count in kept_combos:
    # Get indices for this combo
    mask = np.all(combined_buttons == np.array(combo), axis=1)
    combo_sticks = main_stick[mask]

    # Scale the exclude threshold - smaller groups get more lenient cutoff
    scale_factor = np.sqrt(n_frames / combo_count)
    scaled_exclude_pct = min(
        exclude_pct * scale_factor * 100, 50)  # As percentage, cap at 50%

    # Bucket sticks using polar coordinates
    labels, polar_info = bucket_sticks_polar(
        combo_sticks, n_log_radius=n_log_radius, n_angle=n_angle)

    # Count buckets and find how many needed for coverage
    counter = collections.Counter(labels)
    sorted_buckets = counter.most_common()  # (label, count) sorted by count desc
    bucket_sizes = [count for _, count in sorted_buckets]
    bucket_cumsum = np.cumsum(bucket_sizes)
    points_needed = (1 - scaled_exclude_pct / 100) * combo_count
    n_kept = np.searchsorted(bucket_cumsum, points_needed) + 1
    n_kept = min(n_kept, len(bucket_sizes))

    # Get the labels of the kept buckets
    kept_labels = set(label for label, _ in sorted_buckets[:n_kept])

    # Count kept buckets per radius level
    # r0 = origin (0 or 1), r1..rN = actual radius buckets
    n_angle_per_radius = polar_info.n_angle_per_radius
    radius_offsets = np.cumsum([0] + n_angle_per_radius[:-1])
    buckets_per_radius = [1 if 0 in kept_labels else 0]  # r0 = origin
    for r in range(n_log_radius):
      # Labels for radius r are in range [offset+1, offset+n_angle_r]
      start = radius_offsets[r] + 1
      end = start + n_angle_per_radius[r]
      used = sum(1 for lbl in kept_labels if start <= lbl < end)
      buckets_per_radius.append(used)

    total_clusters += n_kept
    combo_results.append(ComboClusterResult(
        combo=combo,
        count=combo_count,
        n_stick_clusters=n_kept,
        scaled_exclude_pct=scaled_exclude_pct,
        buckets_per_radius=buckets_per_radius,
    ))

  return HierarchicalClusterResult(
      exclude_pct=exclude_pct,
      n_log_radius=n_log_radius,
      n_angle=n_angle,
      n_angle_per_radius=n_angle_per_radius,
      n_button_combos=n_button_combos,
      button_coverage=button_coverage,
      total_clusters=total_clusters,
      combo_results=combo_results,
      button_info=info,
  )


def analyze_cluster_sizes(
    data: dict,
    exclude_pcts: list[float] = [0.05, 0.01, 0.005, 0.001],
    n_log_radius_values: list[int] = [3, 4, 5],
    n_angle_values: list[int] = [16, 32, 64],
) -> None:
  """Analyze total cluster counts for different exclusion percentiles and polar configs.

  Args:
      data: Dict with 'buttons', 'shoulder', 'main_stick'.
      exclude_pcts: List of exclusion fractions to try.
      n_log_radius_values: List of log-radius bucket counts to try.
      n_angle_values: List of angle bucket counts to try.
  """
  results = []

  button_data = get_button_data(data)

  for exclude_pct, n_log_r, n_ang in itertools.product(exclude_pcts, n_log_radius_values, n_angle_values):
    polar_config = f"({n_log_r}, {n_ang})"

    print("\n" + "="*60)
    print(
        f"HIERARCHICAL CLUSTERING (exclude={exclude_pct*100:.2f}%, polar={polar_config})")
    print("="*60)

    result = cluster_hierarchical(
        data, exclude_pct=exclude_pct,
        n_log_radius=n_log_r, n_angle=n_ang,
        button_data=button_data)
    results.append(result)

    pct_str = f"{exclude_pct*100:.2f}%"
    print(f"\nExclude {pct_str}:")
    print(f"  Button combos: {result.n_button_combos} "
          f"(coverage: {result.button_coverage*100:.2f}%)")
    print(f"  Total clusters (button + main stick): {result.total_clusters}")

    # Table of all button combos with buckets per radius
    # r0 = origin, r1..rN = actual radius buckets
    radius_headers = " ".join(f"r{i}" for i in range(n_log_r + 1))
    print(f"\n  {'Button Combo':<30} {'Frames':>10} {'Scaled %':>10} {'Clusters':>10} | {radius_headers}")
    print("  " + "-" * (62 + 3 + 3 * (n_log_r + 1)))
    for cr in result.combo_results:
      combo_str = format_button_combo(cr.combo, result.button_info)
      buckets_str = " ".join(f"{b:>2}" for b in cr.buckets_per_radius)
      print(f"  {combo_str:<30} {cr.count:>10,} {cr.scaled_exclude_pct:>9.2f}% {cr.n_stick_clusters:>10} | {buckets_str}")

  # Summary table
  print("\n" + "-"*50)
  print("Summary:")
  print(f"{'Exclude %':>12} {'Btns':>6} {'(logR,ang)':>12} {'Clusters':>10}")
  print("-"*50)
  for r in results:
    pct_str = f"{r.exclude_pct*100:.2f}%"
    polar_str = f"({r.n_log_radius}, {r.n_angle})"
    print(
        f"{pct_str:>12} {r.n_button_combos:>6} {polar_str:>12} {r.total_clusters:>10}")
