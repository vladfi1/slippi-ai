#!/usr/bin/env python3

import os
import pickle

from absl import app, flags

from slippi_ai import saving

SRC = flags.DEFINE_string(
    'src', 'pickled_models',
    'Path to the directory containing the models to strip')
DST = flags.DEFINE_string(
    'dst', 'stripped_models',
    'Path to the directory to save the stripped models')
VERBOSE = flags.DEFINE_bool(
    'verbose', False, 'Prints out the models that are stripped')

def strip_state(tf_state: dict) -> dict:
  keys = ['policy']
  return {k: tf_state[k] for k in keys}

def needs_copy(src, dst):
  if not os.path.exists(dst):
    return True

  src_time = os.path.getmtime(src)
  dst_time = os.path.getmtime(dst)

  return src_time > dst_time

def run(src: str, dst: str, verbose: bool = False):
  for dirpath, dirnames, filenames in os.walk(src):
    rel_dir = os.path.relpath(dirpath, src)
    dst_dir = os.path.join(dst, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for filename in filenames:
      src_path = os.path.join(dirpath, filename)
      dst_path = os.path.join(dst_dir, filename)

      if not needs_copy(src_path, dst_path):
        continue

      combined_state = saving.load_state_from_disk(src_path)
      combined_state['state'] = strip_state(combined_state['state'])

      with open(dst_path, 'wb') as f:
        pickle.dump(combined_state, f)

      if verbose:
        rel_path = os.path.relpath(src_path, src)
        print(f'Stripped {rel_path}')

def main(_):
  run(SRC.value, DST.value, VERBOSE.value)

if __name__ == '__main__':
  app.run(main)
