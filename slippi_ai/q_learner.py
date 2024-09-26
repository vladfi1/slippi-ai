import dataclasses
from typing import List, Optional

import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.policies import Policy, RecurrentState
from slippi_ai import q_function as q_lib
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
  reward_halflife: float = 8

# TODO: should this be a snt.Module?
class Learner:

  def __init__(self,
      learning_rate: float,
      compile: bool,
      reward_halflife: float,
      q_function: q_lib.QFunction,
      # policy: Policy,
  ):
    # self.policy = policy
    self.q_function = q_function
    # self.policy_optimizer = snt.optimizers.Adam(learning_rate)
    self.q_optimizer = snt.optimizers.Adam(learning_rate)
    self.discount = 0.5 ** (1 / (reward_halflife * 60))
    self.compiled_step = tf.function(self.step) if compile else self.step

  def initial_state(self, batch_size: int) -> RecurrentState:
    return (
        # self.policy.initial_state(batch_size),
        self.q_function.initial_state(batch_size),
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
        self.initial_state(batch_size),
        initial_states)
    q_initial_states, = initial_states
    del initial_states

    # switch axes to time-major
    tm_frames: Frames = tf.nest.map_structure(
        swap_axes, bm_frames)

    with tf.GradientTape() as tape:
      # TODO: take into account delay
      q_outputs, q_final_states = self.q_function.loss(
          tm_frames, q_initial_states, self.discount)

      if train:
        q_params = self.q_function.trainable_variables
        tf_utils.assert_same_variables(tape.watched_variables(), q_params)
        q_grads = tape.gradient(q_outputs.loss, q_params)
        self.q_optimizer.apply(q_grads, q_params)

    final_states = (q_final_states,)
    metrics = dict(
        total_loss=q_outputs.loss,
        # policy=policy_metrics,
        q_function=q_outputs.metrics,
    )

    return metrics, final_states
