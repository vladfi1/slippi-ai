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

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
  )

  def __init__(self,
      embed_game: embed.Embedding,
      learning_rate: float,
      network: snt.Module,
      network_name: str,
      **config):
    self.embed_game = embed_game
    self.network = network
    self.network_name = network_name
    self.controller_head = snt.Linear(embed.embed_controller.size)
    self.optimizer = snt.optimizers.Adam(learning_rate)
    print(f'\nUsing network: {self.network_name}')

    # self.compiled_step = tf.function(self.step)
    self.compiled_step = self.step

  def step(self, batch, initial_states, train=True):
    gamestate, restarting = tf.nest.map_structure(to_time_major, batch)

    flat_gamestate = self.embed_game(gamestate)
    p1_controller = gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    with tf.GradientTape() as tape:
      outputs, final_states = self.network.unroll(
          flat_gamestate, initial_states, restarting)
      controller_prediction = self.controller_head(outputs[:-1])
      next_action_distances = embed.embed_controller.distance(
          controller_prediction, next_action)
      mean_distances = tf.nest.map_structure(
          tf.reduce_mean, next_action_distances)
      loss = tf.add_n(tf.nest.flatten(mean_distances))

    if train:
      params = tape.watched_variables()
      grads = tape.gradient(loss, params)
      self.optimizer.apply(grads, params)
    return loss, final_states
