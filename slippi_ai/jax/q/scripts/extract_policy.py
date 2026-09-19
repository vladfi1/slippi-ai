#!/usr/bin/env python3

import os
import pickle

from absl import app, flags

from slippi_ai import saving

SRC = flags.DEFINE_string(
    'src', 'models/q/policy',
    'Model file to strip, or a directory of them to strip recursively.')
DST = flags.DEFINE_string(
    'dst', 'stripped_models/jax/q_policy',
    'Where to save the stripped models: a file path if --src is a file, '
    'otherwise a directory mirroring --src.')
VERBOSE = flags.DEFINE_bool(
    'verbose', False, 'Prints out the models that are stripped')

def extract_q_policy(q_policy_state: dict) -> dict:
  jax_state = {
    'policy': q_policy_state['state']['q_policy'],
  }

  return {
    'state': jax_state,
    'config': q_policy_state['imitation_config'],
    'name_map': q_policy_state['name_map'],
  }

def needs_copy(src, dst):
  if not os.path.exists(dst):
    return True

  src_time = os.path.getmtime(src)
  dst_time = os.path.getmtime(dst)

  return src_time > dst_time

def extract_file(src_path: str, dst_path: str) -> bool:
  """Strips src_path into dst_path; returns whether anything was written."""
  if not needs_copy(src_path, dst_path):
    return False

  combined_state = saving.load_state_from_disk(src_path)
  combined_state = extract_q_policy(combined_state)

  dst_dir = os.path.dirname(dst_path)
  if dst_dir:
    os.makedirs(dst_dir, exist_ok=True)

  with open(dst_path, 'wb') as f:
    pickle.dump(combined_state, f)

  return True

def run(src: str, dst: str, verbose: bool = False):
  if os.path.isfile(src):
    if os.path.isdir(dst):
      raise ValueError(
          f'--src {src} is a file, so --dst {dst} must be a file path, '
          'but it is an existing directory.')
    if extract_file(src, dst) and verbose:
      print(f'Stripped {src} -> {dst}')
    return

  for dirpath, dirnames, filenames in os.walk(src):
    rel_dir = os.path.relpath(dirpath, src)
    dst_dir = os.path.join(dst, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for filename in filenames:
      src_path = os.path.join(dirpath, filename)
      dst_path = os.path.join(dst_dir, filename)

      if extract_file(src_path, dst_path) and verbose:
        rel_path = os.path.relpath(src_path, src)
        print(f'Stripped {rel_path}')

def main(_):
  run(SRC.value, DST.value, VERBOSE.value)

if __name__ == '__main__':
  app.run(main)
