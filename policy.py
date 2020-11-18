import sonnet as snt
import tensorflow as tf

import embed

class Policy(snt.Module):
  embed_controller = embed.embed_controller

  def __init__(self, network):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = snt.Linear(self.embed_controller.size)
    self.initial_state = self.network.initial_state

  def unroll(self, gamestate, initial_states):
    outputs, final_states = self.network.unroll(
        gamestate, initial_states)
    controller_prediction = self.controller_head(outputs)
    return controller_prediction, final_states

  def loss(self, gamestate, initial_states):
    controller_prediction, final_states = self.unroll(
        gamestate, initial_states)

    p1_controller = gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    next_action_distances = self.embed_controller.distance(
        controller_prediction[:-1], next_action)
    mean_distances = tf.nest.map_structure(
        tf.reduce_mean, next_action_distances)
    loss = tf.add_n(tf.nest.flatten(mean_distances))

    return loss, final_states

  def sample(self, gamestate, initial_state):
    output, final_state = self.network.step(gamestate, initial_state)
    controller_prediction = self.controller_head(output)
    controller_sample = self.embed_controller.sample(controller_prediction)
    return controller_sample, final_state
