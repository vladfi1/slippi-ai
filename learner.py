import sonnet as snt
import tensorflow as tf

from policy import Policy

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class Learner:

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
  )

  def __init__(self,
      learning_rate: float,
      policy: Policy):
    self.policy = policy
    self.optimizer = snt.optimizers.Adam(learning_rate)
    self.compiled_step = tf.function(self.step)

  def step(self, batch, initial_states, train=True):
    gamestate, restarting = tf.nest.map_structure(to_time_major, batch)

    with tf.GradientTape() as tape:
      loss, final_states = self.policy.loss(
          gamestate, restarting, initial_states)

    if train:
      params = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)
    return loss, final_states
