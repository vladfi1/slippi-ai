import sonnet as snt
import tensorflow as tf

from controller_heads import ControllerHead
import embed

class Policy(snt.Module):

  def __init__(self, network, controller_head: ControllerHead):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.initial_state = self.network.initial_state

  def loss(self, gamestate, initial_state):
    output, final_state = self.network.unroll(gamestate, initial_state)

    p1_controller = gamestate['player'][1]['controller_state']
    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    next_action_distances = self.controller_head.log_prob(
        output[:-1], prev_action, next_action)
    mean_distances = tf.nest.map_structure(
        tf.reduce_mean, next_action_distances)
    loss = tf.add_n(tf.nest.flatten(mean_distances))

    return loss, final_state

  def sample(self, gamestate, initial_state):
    output, final_state = self.network.step(gamestate, initial_state)
    p1_controller = gamestate['player'][1]['controller_state']
    controller_sample = self.controller_head.sample(
        output, p1_controller)
    return controller_sample, final_state
