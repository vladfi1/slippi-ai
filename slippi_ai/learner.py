import dataclasses
from typing import List, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai import value_function as vf_lib
from slippi_ai import tf_utils

def swap_axes(t, axis1=0, axis2=1):
  permutation = list(range(len(t.shape)))
  permutation[axis2] = axis1
  permutation[axis1] = axis2
  return tf.transpose(t, permutation)

@dataclasses.dataclass
class LearnerConfig:
  learning_rate: float = 1e-4
  compile: bool = True
  jit_compile: bool = True
  decay_rate: float = 0.
  value_cost: float = 0.5
  reward_halflife: float = 4


# TODO: should this be a snt.Module?
class Learner:

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      value_cost: float,
      reward_halflife: float,
      value_function: Optional[vf_lib.ValueFunction] = None,
      decay_rate: Optional[float] = None,
      jit_compile: bool = True,
  ):
    self.policy = policy
    self.value_function = value_function or vf_lib.FakeValueFunction()
    self.policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.value_optimizer = snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.value_cost = value_cost
    self.discount = 0.5 ** (1 / (reward_halflife * 60))

    self.compile = compile
    self._compiled_step = tf.function(
        self._step, jit_compile=jit_compile, autograph=False)

  def initial_state(self, batch_size: int) -> RecurrentState:
    return (
        self.policy.initial_state(batch_size),
        self.value_function.initial_state(batch_size),
    )

  def _step(
      self,
      bm_frames: Frames,
      # batch: Batch,
      initial_states: RecurrentState,
      train: bool = True,
  ):
    policy_initial_states, value_initial_states = initial_states
    del initial_states

    # switch axes to time-major
    tm_frames: Frames = tf.nest.map_structure(
        swap_axes, bm_frames)

    with tf.GradientTape() as tape:
      policy_loss, policy_final_states, policy_metrics = self.policy.imitation_loss(
          tm_frames, policy_initial_states,
          self.value_cost, self.discount)

      if train:
        policy_params = self.policy.trainable_variables
        tf_utils.assert_same_variables(tape.watched_variables(), policy_params)
        policy_grads = tape.gradient(policy_loss, policy_params)
        self.policy_optimizer.apply(policy_grads, policy_params)

    with tf.GradientTape() as tape:
      # Drop the delayed frames from the value function.
      delay = self.policy.delay
      value_frames = tf.nest.map_structure(
          lambda t: t[:t.shape[0]-delay], tm_frames)
      value_outputs, value_final_states = self.value_function.loss(
          value_frames, value_initial_states, self.discount)

      if train:
        value_params = self.value_function.trainable_variables
        tf_utils.assert_same_variables(tape.watched_variables(), value_params)
        value_grads = tape.gradient(value_outputs.loss, value_params)
        self.value_optimizer.apply(value_grads, value_params)

    if train and self.decay_rate:
      for param in policy_params + value_params:
        param.assign((1 - self.decay_rate) * param)

    final_states = (policy_final_states, value_final_states)
    metrics = dict(
        total_loss=policy_loss + value_outputs.loss,
        policy=policy_metrics,
        value=value_outputs.metrics,
    )

    # convert metrics to batch-major
    # metrics: dict = tf.nest.map_structure(
    #   lambda t: swap_axes(t) if len(t.shape) >= 2 else t,
    #   metrics)

    return metrics, final_states

  def step(
      self,
      frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      compile: Optional[bool] = None,
  ):
    compile = compile if compile is not None else self.compile
    step = self._compiled_step if compile else self._step

    return step(frames, initial_states, train=train)
