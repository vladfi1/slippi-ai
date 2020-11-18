import datetime
import os
import secrets

import utils

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  today = datetime.date.today()
  expt_tag = f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'
  expt_dir = f'experiments/{expt_tag}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir

class TrainManager:

  def __init__(self, learner, data_source, step_kwargs={}):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs

    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self):
    with self.data_profiler:
      batch = next(self.data_source)
    with self.step_profiler:
      loss, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    return loss
