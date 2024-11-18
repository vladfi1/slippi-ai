#!/usr/bin/env python3

import os
import pickle

from absl import app, flags

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
  os.makedirs(dst, exist_ok=True)
  models = os.listdir(src)

  for model in models:
    src_path = os.path.join(src, model)
    dst_path = os.path.join(dst, model)

    if not needs_copy(src_path, dst_path):
      continue

    with open(src_path, 'rb') as f:
      combined_state = pickle.load(f)

    combined_state['state'] = strip_state(combined_state['state'])

    with open(dst_path, 'wb') as f:
      pickle.dump(combined_state, f)

    if verbose:
      print(f'Stripped {model}')

def main(_):
  run(SRC.value, DST.value, VERBOSE.value)

if __name__ == '__main__':
  app.run(main)
