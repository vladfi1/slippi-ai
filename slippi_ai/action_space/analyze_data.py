"""Analyze controller input distributions from Slippi replays."""

import argparse
import os

from slippi_ai import paths

from slippi_ai.action_space.clustering import (
    analyze_buttons,
    analyze_cluster_sizes,
    analyze_shoulder,
    analyze_stick,
    analyze_stick_deltas,
    analyze_stick_polar,
    extract_controllers_from_replays,
    sample_replays_by_character,
)

ANALYSIS_TYPES = {
    'buttons',
    'shoulder',
    'main_stick',
    'c_stick',
    'main_stick_polar',
    'c_stick_polar',
    'main_stick_deltas',
    'c_stick_deltas',
    'hierarchical',
}


def main():
  parser = argparse.ArgumentParser(
      description='Analyze controller input distributions.')
  parser.add_argument('--toy_data', action='store_true',
                      help='Use small toy dataset for quick testing')
  parser.add_argument('--data_dir', default=os.environ.get('DATA_DIR'))
  parser.add_argument('--meta_path', default=os.environ.get('META_PATH'))
  parser.add_argument('--samples_per_character', type=int, default=50)
  parser.add_argument('--max_frames_per_game', type=int, default=1000)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers for extraction')
  parser.add_argument('--analyses', type=str, default=None,
                      help=f'Comma-separated list of analyses to run: '
                           f'{",".join(sorted(ANALYSIS_TYPES))}. '
                           f'If not specified, runs all.')
  # Hierarchical clustering parameters
  parser.add_argument('--exclude_pcts', type=str, default='5,1,0.5,0.1',
                      help='Comma-separated exclusion percentages for hierarchical analysis '
                           '(e.g., "5,1,0.5" means 5%%, 1%%, 0.5%%). Default: 5,1,0.5,0.1')
  parser.add_argument('--n_log_radius', type=str, default='3,4,5',
                      help='Comma-separated log-radius bucket counts. Default: 3,4,5')
  parser.add_argument('--n_angle', type=str, default='16,32,64',
                      help='Comma-separated angle bucket counts. Default: 16,32,64')
  args = parser.parse_args()

  if args.analyses is not None:
    analyses = set(args.analyses.split(','))
    invalid = analyses - ANALYSIS_TYPES
    if invalid:
      parser.error(f"Invalid analysis type(s): {', '.join(sorted(invalid))}. "
                   f"Valid options: {', '.join(sorted(ANALYSIS_TYPES))}")
  else:
    analyses = ANALYSIS_TYPES.copy()

  if not args.data_dir or not args.meta_path:
    raise ValueError("DATA_DIR and META_PATH must be set")

  print(f"Data dir: {args.data_dir}")
  print(f"Meta path: {args.meta_path}")
  print(f"Num workers: {args.num_workers}")

  if args.toy_data:
    meta_path = str(paths.TOY_META_PATH)
    data_dir = str(paths.TOY_DATA_DIR)
    print(f"Using toy dataset at {data_dir}")
  else:
    meta_path = args.meta_path
    data_dir = args.data_dir

  # Sample replays
  sampled = sample_replays_by_character(
      meta_path,
      data_dir,
      samples_per_character=args.samples_per_character,
      seed=args.seed,
  )

  # Extract controller data with button normalization
  print("\nExtracting controller data...")
  print("  - Buttons: X→Y, L→R merged")

  data = extract_controllers_from_replays(
      sampled,
      args.data_dir,
      max_frames_per_game=args.max_frames_per_game,
      normalize_buttons_flag=True,
      num_workers=args.num_workers,
  )

  print(f"\nExtracted {len(data['buttons']):,} frames of controller data")

  # Analyze normalized data
  if 'buttons' in analyses:
    analyze_buttons(data)
  if 'shoulder' in analyses:
    analyze_shoulder(data['shoulder'])
  if 'main_stick' in analyses:
    analyze_stick(data['main_stick'], 'main')
  if 'c_stick' in analyses:
    analyze_stick(data['c_stick'], 'c')
  if 'main_stick_polar' in analyses:
    analyze_stick_polar(data['main_stick'], 'main')
  if 'c_stick_polar' in analyses:
    analyze_stick_polar(data['c_stick'], 'c')
  if 'main_stick_deltas' in analyses:
    analyze_stick_deltas(data['main_stick_deltas'],
                         data['main_stick_delta_dsts'], 'main')
  if 'c_stick_deltas' in analyses:
    analyze_stick_deltas(data['c_stick_deltas'],
                         data['c_stick_delta_dsts'], 'c')
  if 'hierarchical' in analyses:
    # Parse hierarchical clustering parameters
    exclude_pcts = [float(x) / 100 for x in args.exclude_pcts.split(',')]
    n_log_radius_values = [int(x) for x in args.n_log_radius.split(',')]
    n_angle_values = [int(x) for x in args.n_angle.split(',')]
    analyze_cluster_sizes(
        data,
        exclude_pcts=exclude_pcts,
        n_log_radius_values=n_log_radius_values,
        n_angle_values=n_angle_values,
    )


if __name__ == '__main__':
  main()
