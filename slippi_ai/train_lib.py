import datetime
from typing import Optional
import os
import secrets

import numpy as np
import tensorflow as tf
import tree

import wandb

from slippi_ai import (
    data,
    utils,
)

from slippi_ai.learner import Learner

def get_experiment_tag():
  today = datetime.date.today()
  return f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'


class TrainManager:

  def __init__(
      self,
      learner: Learner,
      data_source: data.DataSource,
      step_kwargs={},
  ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.total_frames = 0
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self) -> tuple[dict, data.Batch]:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
      # batch = sanitize_batch(batch)
    with self.step_profiler:
      stats, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    num_frames = batch.frames.state_action.state.stage.size
    self.total_frames += num_frames
    stats.update(
        epoch=epoch,
        num_frames=num_frames,
        total_frames=self.total_frames,
    )
    return stats, batch

def mean(value):
  if isinstance(value, tf.Tensor):
    value = value.numpy()
  if isinstance(value, np.ndarray):
    value = value.mean().item()
  return value

def log_stats(
    stats: tree.Structure,
    step: Optional[int] = None,
    take_mean: bool = True,
):
  if take_mean:
    stats = tree.map_structure(mean, stats)
  wandb.log(data=stats, step=step)
