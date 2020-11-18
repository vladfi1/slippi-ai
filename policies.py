import sonnet as snt
import tensorflow as tf

from controller_heads import ControllerHead
import embed

def get_p1_controller(gamestate, action_repeat):
  p1_controller = gamestate['player'][1]['controller_state']
  return dict(
      controller=p1_controller,
      action_repeat=action_repeat)

class Policy(snt.Module):

  def __init__(self, network, controller_head: ControllerHead):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.initial_state = self.network.initial_state

  def loss(self, gamestate, initial_state):
    # TODO: let network embed action repeat as well
    gamestate, action_repeat = gamestate
    output, final_state = self.network.unroll(gamestate, initial_state)

    p1_controller = get_p1_controller(gamestate, action_repeat)
    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    next_action_distances = self.controller_head.log_prob(
        output[:-1], prev_action, next_action)
    return next_action_distances, final_state

  def sample(self, gamestate, initial_state):
    gamestate, action_repeat = gamestate
    output, final_state = self.network.step(gamestate, initial_state)
    p1_controller = get_p1_controller(gamestate, action_repeat)
    controller_sample = self.controller_head.sample(
        output, p1_controller)
    return controller_sample, final_state
