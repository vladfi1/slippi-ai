from typing import List, Optional
import sonnet as snt
import tensorflow as tf

from slippi_ai.data import Batch
from slippi_ai.policies import Policy
from slippi_ai import embed, utils

# TODO: should this be a snt.Module?
class Learner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
      compile=True,
      decay_rate=0.,
      value_cost=0.5,
      reward_halflife=2,  # measured in seconds
  )

  def __init__(self,
      learning_rate: float,
      compile: bool,
      policy: Policy,
      value_cost: float,
      reward_halflife: float,
      optimizer: Optional[snt.Optimizer] = None,
      decay_rate: Optional[float] = None,
  ):
    self.policy = policy
    self.optimizer = optimizer or snt.optimizers.Adam(learning_rate)
    self.decay_rate = decay_rate
    self.value_cost = value_cost
    self.discount = 0.5 ** (1 / reward_halflife * 60)
    self.compiled_step = tf.function(self.step) if compile else self.step

  def step(self, batch: Batch, initial_states, train=True):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.policy.initial_state(restarting.shape[0]),
        initial_states)

    # switch axes to time-major
    tm_gamestate: embed.StateActionReward = tf.nest.map_structure(
        utils.to_time_major, bm_gamestate)

    with tf.GradientTape() as tape:
      loss, final_states, metrics = self.policy.imitation_loss(
          tm_gamestate, initial_states,
          self.value_cost, self.discount)

    if train:
      params: List[tf.Variable] = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)

      if self.decay_rate:
        for param in params:
          param.assign((1 - self.decay_rate) * param)

    return metrics, final_states
