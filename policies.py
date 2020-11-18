import sonnet as snt
import tensorflow as tf

import embed

class Policy(snt.Module):

  def step(self, gamestate, initial_state):
    return NotImplementedError()

  def unroll(self, gamestate, initial_state):
    return NotImplementedError()

  def loss(self, gamestate, initial_state):
    controller_prediction, final_state = self.unroll(
        gamestate, initial_state)

    p1_controller = gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    next_action_distances = embed.embed_controller.distance(
        controller_prediction[:-1], next_action)
    mean_distances = tf.nest.map_structure(
        tf.reduce_mean, next_action_distances)
    loss = tf.add_n(tf.nest.flatten(mean_distances))

    return loss, final_state

  def sample(self, gamestate, initial_state):
    controller_prediction, final_state = self.step(gamestate, initial_state)
    controller_sample = embed.embed_controller.sample(controller_prediction)
    return controller_sample, final_state


class DefaultPolicy(Policy):

  def __init__(self, network):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = snt.Linear(embed.embed_controller.size)
    self.initial_state = self.network.initial_state

  def step(self, gamestate, initial_state):
    output, final_state = self.network.step(
        gamestate, initial_state)
    controller_prediction = self.controller_head(output)
    return controller_prediction, final_state

  def unroll(self, gamestate, initial_state):
    outputs, final_state = self.network.unroll(
        gamestate, initial_state)
    controller_prediction = self.controller_head(outputs)
    return controller_prediction, final_state

class ResidualPolicy(Policy):
  def __init__(self, network):
    super().__init__(name='ResidualPolicy')
    self.network = network
    self.controller_head = snt.Linear(embed.embed_controller.size)
    '''
      TODO - Consider more efficient [1, 18] element-wise rescaling vector
      instead of [18, 18] linear mat for recurrent head?
    '''
    self.recurrent_head = snt.Linear(embed.embed_controller.size)
    self.initial_state = self.network.initial_state

  def controller_prediction(self, gamestate, network_output):
    '''
      Predict the residual difference between the next and current
      controller state
    '''
    resid_controller_prediction = self.controller_head(network_output)

    p1_controller = gamestate['player'][1]['controller_state']
    prev_action = embed.embed_controller(p1_controller)
    prev_action = self.recurrent_head(prev_action)

    return prev_action + resid_controller_prediction

  def step(self, gamestate, initial_state):
    output, final_state = self.network.step(gamestate, initial_state)
    controller_prediction = self.controller_prediction(gamestate, output)
    return controller_prediction, final_state

  def unroll(self, gamestate, initial_state):
    output, final_state = self.network.unroll(gamestate, initial_state)
    controller_prediction = self.controller_prediction(gamestate, output)
    return controller_prediction, final_state

CONSTRUCTORS = dict(
    default=DefaultPolicy,
    residual=ResidualPolicy,
)

DEFAULT_CONFIG = dict(
    name='default',
)

def construct_policy(name):
  return CONSTRUCTORS[name]
