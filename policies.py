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
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    loss = tf.add_n(tf.nest.flatten(distances))

    return loss, final_state, distances

  def sample(self, gamestate, initial_state):
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed)
    output, final_state = self.network.step(inputs, initial_state)

    controller_sample = self.controller_head.sample(
        output, p1_controller)
    return controller_sample, final_state
