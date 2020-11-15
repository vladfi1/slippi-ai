import sonnet as snt
import tensorflow as tf

import embed
import networks

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class Learner:

  def __init__(self, embed_game, network=networks.DEFAULT_CONFIG):
    self.network = networks.construct_network(**network)
    self.controller_head = snt.Linear(embed.embed_controller.size)
    self.optimizer = snt.optimizers.Adam(1e-4)
    self.embed_game = embed_game

  @tf.function
  def step(self, batch, train=True):
    gamestate, restarting = tf.nest.map_structure(to_time_major, batch)

    flat_gamestate = self.embed_game(gamestate)
    prev_gamestate = flat_gamestate[:-1]

    p1_controller = gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    with tf.GradientTape() as tape:
      outputs = self.network(prev_gamestate)
      controller_prediction = self.controller_head(outputs)
      next_action_distances = embed.embed_controller.distance(
          controller_prediction, next_action)
      mean_distances = tf.nest.map_structure(
          tf.reduce_mean, next_action_distances)
      loss = tf.add_n(tf.nest.flatten(mean_distances))

    if train:
      params = self.network.trainable_variables
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)
    return loss
