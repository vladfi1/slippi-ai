from typing import Any, Sequence, Tuple
import sonnet as snt
import tensorflow as tf

from controller_heads import ControllerHead
import data
import networks

RecurrentState = networks.RecurrentState
ControllerWithRepeat = dict

def get_p1_controller(
    gamestates: data.Game,
    action_repeat: Sequence[int],
) -> ControllerWithRepeat:
  p1_controller = gamestates['player'][1]['controller_state']
  return dict(
      controller=p1_controller,
      action_repeat=action_repeat)

class Policy(snt.Module):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.initial_state = self.network.initial_state

  def loss(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    gamestates = compressed.states
    action_repeat = compressed.counts
    p1_controller = get_p1_controller(gamestates, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestates, p1_controller_embed)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    loss = tf.add_n(tf.nest.flatten(distances))

    return loss, final_state, distances

  def sample(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[ControllerWithRepeat, RecurrentState]:
    gamestates = compressed.states
    action_repeat = compressed.counts
    p1_controller = get_p1_controller(gamestates, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestates, p1_controller_embed)
    output, final_state = self.network.step(inputs, initial_state)

    controller_sample = self.controller_head.sample(
        output, p1_controller, **kwargs)
    return controller_sample, final_state
