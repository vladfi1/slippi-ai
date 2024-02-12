import datetime
from typing import Optional
import os
import secrets

import numpy as np
import tensorflow as tf
import tree

import sacred

from slippi_ai import (
    data,
    utils,
)

from slippi_ai.learner import Learner

def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  expt_dir = f'experiments/{get_experiment_tag()}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir


class TrainManager:

  def __init__(
      self,
      learner: Learner,
      data_source: data.DataSource,
      step_kwargs={},
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_states = [
        learner.policy.initial_state(1)
        for _ in range(data_source.game_buffer)]
    self.step_kwargs = step_kwargs
    self.total_frames = 0
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self) -> dict:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
      # Assemble initial states.
      # TODO: do this in TF with scatter_update?
      initial_states = [self.hidden_states[i] for i in batch.index]
      initial_state = utils.batch_nest(initial_states)
    with self.step_profiler:
      stats, final_state = self.learner.compiled_step(
          batch, initial_state, **self.step_kwargs)
      # Now update hidden states.
      for j, i in enumerate(batch.index):
        self.hidden_states[i] = tf.nest.map_structure(
            lambda t: t[j], final_state)
    num_frames = np.prod(batch.frames.state_action.state.stage.shape[-1])
    self.total_frames += num_frames
    stats.update(
        epoch=epoch,
        num_frames=num_frames,
        total_frames=self.total_frames,
        meta=batch.meta,
    )
    return stats

def log_stats(
    ex: sacred.Experiment,
    stats,
    step: Optional[int] = None,
    sep: str ='.',
    take_mean: bool = True,
):
  def log(path, value):
    if isinstance(value, tf.Tensor):
      value = value.numpy()
    if isinstance(value, np.ndarray):
      if take_mean:
        if issubclass(value.dtype.type, np.floating):
          value = value.mean().item()
        else:
          return
      else:
        # The MongoObserver doesn't like numpy types for some reason.
        value = value.tolist()
    key = sep.join(map(str, path))
    ex.log_scalar(key, value, step=step)
  tree.map_structure_with_path(log, stats)
