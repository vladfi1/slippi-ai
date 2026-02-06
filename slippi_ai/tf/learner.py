import dataclasses
import functools
from typing import List, Optional

import numpy as np
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch, Frames
from slippi_ai.tf.policies import Policy, RecurrentState
from slippi_ai.tf import value_function as vf_lib
from slippi_ai import utils
from slippi_ai.tf import tf_utils

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
  minibatch_size: int = 0  # 0 means no minibatching

# TODO: should this be a snt.Module?
class Learner:

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      value_cost: float,
      reward_halflife: float,
      minibatch_size: int = 0,
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
    self.minibatch_size = minibatch_size

    self.compile = compile

    compile_kwargs = dict(jit_compile=jit_compile, autograph=False)

    self._compiled_step = tf.function(
        self._step, **compile_kwargs)
    self._compiled_step_grads_acc = tf.function(
        self._step_grads_acc, **compile_kwargs)

  def initial_state(self, batch_size: int) -> RecurrentState:
    return (
        self.policy.initial_state(batch_size),
        self.value_function.initial_state(batch_size),
    )

  @property
  def policy_vars(self) -> List[tf.Variable]:
    return self.policy.trainable_variables

  @property
  def value_vars(self) -> List[tf.Variable]:
    return self.value_function.trainable_variables

  def _step_grads(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      apply_grads: bool = True,
  ) -> tuple[dict, RecurrentState, tuple[list[tf.Tensor], list[tf.Tensor]]]:
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
        policy_params = self.policy_vars
        tf_utils.assert_same_variables(tape.watched_variables(), policy_params)
        policy_grads = tape.gradient(policy_loss, policy_params)
        if apply_grads:
          self.policy_optimizer.apply(policy_grads, policy_params)
      else:
        policy_grads = []

    with tf.GradientTape() as tape:
      # Drop the delayed frames from the value function.
      delay = self.policy.delay
      value_frames = tf.nest.map_structure(
          lambda t: t[:t.shape[0]-delay], tm_frames)
      value_outputs, value_final_states = self.value_function.loss(
          value_frames, value_initial_states, self.discount)

      if train:
        value_params = self.value_vars
        tf_utils.assert_same_variables(tape.watched_variables(), value_params)
        value_grads = tape.gradient(value_outputs.loss, value_params)
        if apply_grads:
          self.value_optimizer.apply(value_grads, value_params)
      else:
        value_grads = []

    final_states = (policy_final_states, value_final_states)
    metrics = dict(
        total_loss=policy_loss + value_outputs.loss,
        policy=policy_metrics,
        value=value_outputs.metrics,
    )

    grads = (policy_grads, value_grads)

    # convert metrics to batch-major
    # metrics: dict = tf.nest.map_structure(
    #   lambda t: swap_axes(t) if len(t.shape) >= 2 else t,
    #   metrics)

    return metrics, final_states, grads

  def _step_grads_acc(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      grads_acc: tuple[list[tf.Tensor], list[tf.Tensor]],
      train: bool = True,
  ):
    metrics, final_states, grads = self._step_grads(
        bm_frames, initial_states, train=train, apply_grads=False)

    if train:
      grads_acc = utils.map_nt(
          lambda a, g: a + g if g is not None else a,
          grads_acc, grads)

    return metrics, final_states, grads_acc

  @tf.function
  def apply_grads(self, grads: tuple[list[tf.Tensor], list[tf.Tensor]], scale: float = 1.0):
    grads = utils.map_nt(
        lambda g: g * scale if g is not None else g, grads)
    policy_grads, value_grads = grads
    self.policy_optimizer.apply(policy_grads, self.policy_vars)
    self.value_optimizer.apply(value_grads, self.value_vars)

  def _step(
      self,
      bm_frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
  ):
    metrics, final_states, _ = self._step_grads(
        bm_frames, initial_states, train=train, apply_grads=True)

    return metrics, final_states

  @tf.function
  def combine_final_states(
      self,
      final_states_list: list[RecurrentState],
  ) -> RecurrentState:
    return tf.nest.map_structure(
        lambda *args: tf.concat(args, axis=0),
        *final_states_list)

  @tf.function
  def combine_metrics(
      self,
      metrics_list: list[dict],
  ) -> dict:
    # Stats are either scalars or (time, batch)-shaped.
    def _combine_stats(*args: tf.Tensor) -> tf.Tensor:
      t = args[0]
      if len(t.shape) == 0:
        return tf.reduce_mean(tf.stack(args, axis=0), axis=0)
      else:
        return tf.concat(args, axis=1)  # concat along batch dimension
    return tf.nest.map_structure(_combine_stats, *metrics_list)

  @functools.cached_property
  def zero_grads(self):
    return tf.nest.map_structure(
        tf.zeros_like, (self.policy_vars, self.value_vars))

  def step(
      self,
      frames: Frames,
      initial_states: RecurrentState,
      train: bool = True,
      compile: Optional[bool] = None,
  ) -> tuple[dict, RecurrentState]:
    compile = compile if compile is not None else self.compile

    if self.minibatch_size == 0:
      step = self._compiled_step if compile else self._step
      return step(
          frames,
          initial_states,
          train=train,
      )

    step_grads_acc = self._compiled_step_grads_acc if compile else self._step_grads_acc

    batch_size = frames.is_resetting.shape[0]
    minibatch_size = self.minibatch_size

    num_splits, r = divmod(batch_size, minibatch_size)
    if r != 0:
      raise ValueError(f"Batch size {batch_size} is not divisible by minibatch size {minibatch_size}")

    if train:
      grads_acc = self.zero_grads
    else:
      grads_acc = ([], [])

    final_states_list = []
    metrics_list = []

    for i in range(num_splits):
      start = i * minibatch_size
      end = start + minibatch_size
      slice_mb = lambda t: t[start:end]

      bm_frames_mb = utils.map_nt(slice_mb, frames)
      initial_states_mb = utils.map_nt(slice_mb, initial_states)

      metrics_mb, final_states_mb, grads_acc = step_grads_acc(
          bm_frames_mb,
          initial_states_mb,
          train=train,
          grads_acc=grads_acc,
      )

      final_states_list.append(final_states_mb)
      metrics_list.append(metrics_mb)

    if train:
      self.apply_grads(grads_acc, scale=1.0 / num_splits)

    final_states = self.combine_final_states(final_states_list)
    metrics = self.combine_metrics(metrics_list)

    return metrics, final_states
