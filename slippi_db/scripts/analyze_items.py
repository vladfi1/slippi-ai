#!/usr/bin/env python3
"""
Analyze item statistics from Slippi replay archives.

This script traverses .zip archives containing Slippi replay files and computes
statistics on items (how many items of each type/state appear).
"""

import concurrent.futures
import json
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from typing import Dict, Optional

import melee
import numpy as np
import peppi_py
from absl import app, flags, logging
from tqdm import tqdm

from slippi_db import utils

# Command line flags
INPUT = flags.DEFINE_string('input', None, 'Input archive to analyze (.zip file).', required=True)
OUTPUT = flags.DEFINE_string('output', 'item_stats.json', 'Output JSON file for statistics.')
NUM_THREADS = flags.DEFINE_integer('threads', 1, 'Number of threads to use for processing.')
IN_MEMORY = flags.DEFINE_bool('in_memory', True, 'Use in-memory temporary files for processing.')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 30, 'Interval in seconds to log progress.')
LIMIT = flags.DEFINE_integer('limit', None, 'Limit number of files to process (for testing).')


def _nested_defaultdict():
  """Helper function to create nested defaultdict (pickle-able)."""
  return defaultdict(int)

def frame_to_string(frame: int) -> str:
    """Convert a frame number to a string representation."""
    digits = []
    for factor in [60, 60, 60]:
        digits.append(frame % factor)
        frame //= factor
    return ":".join(map(str, reversed(digits)))

class ItemStats:
  """Accumulator for item statistics."""

  def __init__(self):
    self.total_games = 0
    self.total_frames = 0
    self.games_with_items = 0
    self.item_type_counts = defaultdict(int)
    self.item_state_counts = defaultdict(int)
    self.item_type_state_counts = defaultdict(_nested_defaultdict)
    self.items_per_game = []
    self.max_items_per_frame = 0
    self.unknown_item_examples = {}  # Maps unknown item type to first occurrence info
    self.current_game_file = None  # Track current game file being processed
    self.archive_path = None  # Track source archive path

  def add_game(self, peppi_game: peppi_py.Game, game_file: str = None):
    """Process a single game and update statistics."""
    self.total_games += 1
    self.current_game_file = game_file

    frames = peppi_game.frames
    if frames is None:
      return

    self.total_frames += len(frames.id)

    if frames.items is None:
      return

    game_has_items = False
    items_in_game = 0
    game_max_items_per_frame = 0

    for frame_idx, frame_items in zip(frames.id, frames.items):
      # Get item types and states for this frame
      item_types = frame_items.type.to_numpy()
      item_states = frame_items.state.to_numpy()

      # Count items in this frame
      items_in_frame = len(item_types)
      game_max_items_per_frame = max(game_max_items_per_frame, items_in_frame)

      # Process each item slot
      for item_type, item_state in zip(item_types, item_states):
        game_has_items = True
        items_in_game += 1

        # Get item name using melee.enums.ProjectileType
        try:
          projectile_type = melee.enums.ProjectileType(item_type)
          item_name = projectile_type.name
        except ValueError:
          item_name = f'Unknown {item_type}'
          # Collect first example of each unknown item type
          if item_type not in self.unknown_item_examples:
            self.unknown_item_examples[item_type] = {
              'item_type': hex(int(item_type)),
              'item_state': hex(int(item_state)),
              'game_file': self.current_game_file,
              'frame_index': frame_idx,
              'game_time': frame_to_string(frame_idx),
              'item_name': item_name
            }

        # Update counts
        self.item_type_counts[item_name] += 1
        self.item_state_counts[int(item_state)] += 1
        self.item_type_state_counts[item_name][int(item_state)] += 1

    if game_has_items:
      self.games_with_items += 1
      self.items_per_game.append(items_in_game)

    # Update max items per frame across all games
    self.max_items_per_frame = max(self.max_items_per_frame, game_max_items_per_frame)

  def merge_from(self, other: 'ItemStats'):
    """Merge statistics from another ItemStats instance."""
    self.total_games += other.total_games
    self.total_frames += other.total_frames
    self.games_with_items += other.games_with_items

    for k, v in other.item_type_counts.items():
      self.item_type_counts[k] += v

    for k, v in other.item_state_counts.items():
      self.item_state_counts[k] += v

    for item_type, states in other.item_type_state_counts.items():
      for state, count in states.items():
        self.item_type_state_counts[item_type][state] += count

    self.items_per_game.extend(other.items_per_game)
    self.max_items_per_frame = max(self.max_items_per_frame, other.max_items_per_frame)

    # Merge unknown item examples (keep first occurrence of each)
    for item_type, example in other.unknown_item_examples.items():
      if item_type not in self.unknown_item_examples:
        self.unknown_item_examples[item_type] = example

  def to_dict(self) -> Dict:
    """Convert statistics to dictionary for JSON serialization."""
    return {
      'total_games': self.total_games,
      'total_frames': self.total_frames,
      'games_with_items': self.games_with_items,
      'percentage_games_with_items': (
        100.0 * self.games_with_items / self.total_games
        if self.total_games > 0 else 0
      ),
      'item_type_counts': dict(self.item_type_counts),
      'item_state_counts': {str(k): v for k, v in self.item_state_counts.items()},
      'item_type_state_counts': {
        item_type: {str(state): count for state, count in states.items()}
        for item_type, states in self.item_type_state_counts.items()
      },
      'items_per_game_stats': {
        'mean': np.mean(self.items_per_game) if self.items_per_game else 0,
        'median': np.median(self.items_per_game) if self.items_per_game else 0,
        'max': max(self.items_per_game) if self.items_per_game else 0,
        'min': min(self.items_per_game) if self.items_per_game else 0,
      } if self.items_per_game else None,
      'max_items_per_frame': self.max_items_per_frame,
      'unknown_item_examples': list(self.unknown_item_examples.values()),
      'archive_path': self.archive_path,
    }

def process_slp_file(local_file: utils.ZipFile, in_memory: bool) -> Optional[ItemStats]:
  """Process a single SLP file and return its statistics."""
  try:
    tmp_parent_dir = utils.get_tmp_dir(in_memory=in_memory)

    with tempfile.TemporaryDirectory(dir=tmp_parent_dir) as tmp_dir:
      # Read and decompress the file
      raw_data = local_file.read_raw()
      slp_data = local_file.from_raw(raw_data)

      # Write to temporary file
      slp_path = os.path.join(tmp_dir, 'game.slp')
      with open(slp_path, 'wb') as f:
        f.write(slp_data)

      # Parse with peppi
      game = peppi_py.read_slippi(slp_path)

      # Compute statistics
      stats = ItemStats()
      stats.add_game(game, game_file=local_file.path)

      return stats

  except Exception as e:
    logging.warning(f'Error processing {local_file.path}: {e}')
    return None

def analyze_archive(
  input_path: str,
  num_threads: int = 1,
  in_memory: bool = True,
  log_interval: int = 30,
  limit: Optional[int] = None,
) -> ItemStats:
  """Analyze item statistics from a Slippi archive."""

  if not os.path.exists(input_path):
    raise FileNotFoundError(f'Input path does not exist: {input_path}')

  if not input_path.endswith('.zip'):
    raise ValueError(f'Input path must be a .zip file: {input_path}')

  # Get absolute path
  absolute_path = os.path.abspath(input_path)

  # Get list of SLP files in archive
  logging.info(f'Opening archive: {input_path}')
  zf = zipfile.ZipFile(input_path, 'r')

  slp_files = []
  for zip_info in zf.infolist():
    if zip_info.is_dir():
      continue

    if utils.is_slp_file(zip_info.filename):
      slp_files.append(utils.ZipFile(input_path, zip_info.filename))

      if limit and len(slp_files) >= limit:
        break

  zf.close()

  logging.info(f'Found {len(slp_files)} SLP files in archive')

  # Process files
  combined_stats = ItemStats()
  combined_stats.archive_path = absolute_path

  if num_threads == 1:
    # Single-threaded processing
    with tqdm(total=len(slp_files), desc='Processing files') as pbar:
      for slp_file in slp_files:
        stats = process_slp_file(slp_file, in_memory)
        if stats:
          combined_stats.merge_from(stats)
        pbar.update(1)
  else:
    # Multi-threaded processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
      futures = []
      for slp_file in slp_files:
        futures.append(executor.submit(process_slp_file, slp_file, in_memory))

      with tqdm(total=len(futures), desc='Processing files') as pbar:
        last_log_time = time.time()

        for future in concurrent.futures.as_completed(futures):
          stats = future.result()
          if stats:
            combined_stats.merge_from(stats)
          pbar.update(1)

          # Log progress periodically
          if time.time() - last_log_time > log_interval:
            last_log_time = time.time()
            completion_rate = pbar.n / len(futures)
            logging.info(f'Progress: {100 * completion_rate:.1f}%')

  return combined_stats

def main(_):
  """Main entry point."""

  # Analyze the archive
  stats = analyze_archive(
    input_path=INPUT.value,
    num_threads=NUM_THREADS.value,
    in_memory=IN_MEMORY.value,
    log_interval=LOG_INTERVAL.value,
    limit=LIMIT.value,
  )

  # Convert to dictionary
  stats_dict = stats.to_dict()

  # Print summary
  print("\n=== Item Statistics Summary ===")
  print(f"Total games analyzed: {stats_dict['total_games']}")
  print(f"Games with items: {stats_dict['games_with_items']} ({stats_dict['percentage_games_with_items']:.1f}%)")
  print(f"Total frames: {stats_dict['total_frames']}")
  print(f"Max items on any frame: {stats_dict['max_items_per_frame']}")

  if stats_dict['item_type_counts']:
    print("\n=== Top 10 Item Types ===")
    sorted_items = sorted(stats_dict['item_type_counts'].items(), key=lambda x: x[1], reverse=True)
    for i, (item_type, count) in enumerate(sorted_items[:10], 1):
      print(f"{i}. {item_type}: {count:,}")

  if stats_dict['items_per_game_stats']:
    print("\n=== Items per Game Statistics ===")
    ips = stats_dict['items_per_game_stats']
    print(f"Mean: {ips['mean']:.1f}")
    print(f"Median: {ips['median']:.1f}")
    print(f"Min: {ips['min']}")
    print(f"Max: {ips['max']}")

  if stats_dict['unknown_item_examples']:
    print("\n=== Unknown Item Types Found ===")
    print(f"Found {len(stats_dict['unknown_item_examples'])} unknown item type(s):")
    for example in stats_dict['unknown_item_examples']:
      print(f"  - Type {example['item_type']} (state {example['item_state']})")
      print(f"    First seen in: {example['game_file']}")
      print(f"    At frame index: {example['frame_index']}")

  # Save to JSON
  with open(OUTPUT.value, 'w') as f:
    json.dump(stats_dict, f, indent=2)

  print(f"\nFull statistics saved to {OUTPUT.value}")

if __name__ == '__main__':
  app.run(main)