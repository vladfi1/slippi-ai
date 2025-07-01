#!/usr/bin/env python3

import os
import pickle

from absl import app, flags, logging

PATH = flags.DEFINE_string('path', None, 'Path to the experiment directory')
DELAY = flags.DEFINE_integer('delay', None, 'Delay value to set')

def update_delay(experiment_path: str, delay: int):
  path = os.path.join(experiment_path, "latest.pkl")

  if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist.")

  with open(path, 'rb') as f:
    state = pickle.load(f)

  old_delay = state['config']['policy']['delay']

  if old_delay == delay:
    logging.info(f"Delay is already set to {delay}. No changes made.")
    return

  logging.info(f"Updating delay from {old_delay} to {delay}.")

  state['config']['policy']['delay'] = delay

  old_teacher: str = state['rl_config']['teacher']
  old_delay_str = f'd{old_delay}'
  new_delay_str = f'd{delay}'

  if old_delay_str not in old_teacher:
    raise ValueError(f"'{old_delay_str}' not found in teacher name '{old_teacher}'")

  new_teacher = old_teacher.replace(old_delay_str, new_delay_str)
  if not os.path.exists(new_teacher):
    raise FileNotFoundError(f"New teacher model '{new_teacher}' does not exist.")

  state['rl_config']['teacher'] = new_teacher

  if old_delay_str not in experiment_path:
    raise ValueError(f"'{old_delay_str}' not found in experiment path '{experiment_path}'")

  new_experiment_path = experiment_path.replace(old_delay_str, new_delay_str)
  # new_experiment_path += '_from_' + old_delay_str

  if os.path.exists(new_experiment_path):
    raise FileExistsError(f"New experiment path '{new_experiment_path}' already exists.")

  os.makedirs(new_experiment_path, exist_ok=True)
  new_path = os.path.join(new_experiment_path, "latest.pkl")
  with open(new_path, 'wb') as f:
    pickle.dump(state, f)

  logging.info(f"Saved updated state to {new_path}")

def main(argv):
  del argv  # Unused.

  experiment_path = PATH.value
  delay = DELAY.value

  logging.info(f"Updating delay in {experiment_path} to {delay}")

  update_delay(experiment_path, delay)

if __name__ == '__main__':
  flags.mark_flag_as_required('path')
  flags.mark_flag_as_required('delay')
  app.run(main)
