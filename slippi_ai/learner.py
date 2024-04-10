import dataclasses
from typing import List, Optional

import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai.value_function import ValueFunction

def swap_axes(t, axis1=0, axis2=1):
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return tf.transpose(t, permutation)

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  compile: bool = True
  decay_rate: float = 0.
  value_cost: float = 0.5
  reward_halflife: float = 2
  predict: int = 0

class FakeValueFunction(snt.Module):

  def initial_state(self, batch_size: int) -> RecurrentState:
    del batch_size
    return ()

  def loss(self, frames, initial_state, discount):
    del frames, initial_state, discount
    return tf.constant(0.), (), {}

# TODO: should this be a snt.Module?
class Learner:

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      value_cost: float,
      reward_halflife: float,
      value_function: Optional[ValueFunction] = None,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
      predict: int = 0,
  ):
    self.policy = policy
    self.value_function = value_function or FakeValueFunction()
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.value_cost = value_cost
    self.discount = 0.5 ** (1 / reward_halflife * 60)
    self.compiled_step = tf.function(self.step) if compile else self.step

  def initial_state(self, batch_size: int) -> RecurrentState:
    return (
        self.policy.initial_state(batch_size),
        self.value_function.initial_state(batch_size),
    )

  def step(
      self,
      batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ):
    bm_frames = batch.frames
    restarting = batch.needs_reset
    batch_size = restarting.shape[0]

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.initial_state(restarting.shape[0]),
        initial_states)
    policy_initial_states, value_initial_states = initial_states
    del initial_states

    # switch axes to time-major
    tm_frames: Frames = tf.nest.map_structure(
        swap_axes, bm_frames)

    with tf.GradientTape() as tape:
      policy_loss, policy_final_states, policy_metrics = self.policy.loss(
          tm_frames, policy_initial_states,
          self.value_cost, self.discount)

      # Drop the delayed frames from the value function.
      delay = self.policy.delay
      value_frames = tf.nest.map_structure(
          lambda t: t[:t.shape[0]-delay], tm_frames)
      value_loss, value_final_states, value_metrics = self.value_function.loss(
          value_frames, value_initial_states, self.discount)

      loss = policy_loss + value_loss
      final_states = (policy_final_states, value_final_states)
      metrics = dict(
          policy=policy_metrics,
          value=value_metrics,
          total_loss=loss,
      )

    # convert metrics to batch-major
    # metrics: dict = tf.nest.map_structure(
    #   lambda t: swap_axes(t) if len(t.shape) >= 2 else t,
    #   metrics)

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_variables = (
          self.policy.trainable_variables +
          self.value_function.trainable_variables)
      trainable_names = [v.name for v in trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return metrics, final_states
