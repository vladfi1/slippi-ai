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
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs
    self.total_frames = 0
    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self) -> dict:
    with self.data_profiler:
      batch, epoch = next(self.data_source)
    with self.step_profiler:
      stats, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    num_frames = np.prod(batch.frames.state_action.state.stage.shape)
    self.total_frames += num_frames
    stats.update(
        epoch=epoch,
        num_frames=num_frames,
        total_frames=self.total_frames,
    )
    return stats

def log_stats(
    stats: tree.Structure,
    step: Optional[int] = None,
    sep: str ='.',
):
  def take_mean(value):
    if isinstance(value, tf.Tensor):
      value = value.numpy()
    if isinstance(value, np.ndarray):
      value = value.mean().item()
    return value

  stats = tree.map_structure(take_mean, stats)
  wandb.log(data=stats, step=step)
