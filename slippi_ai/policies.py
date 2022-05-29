from typing import Any, Tuple
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai import networks, embed

RecurrentState = networks.RecurrentState

class Policy(snt.Module):

  def __init__(
      self,
      network: networks.Network,
      controller_head: ControllerHead,
      embed_state_action: embed.StructEmbedding[embed.StateActionReward],
  ):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.embed_state_action = embed_state_action
    self.initial_state = self.network.initial_state

  def loss(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    inputs = self.embed_state_action(state_action)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    action = state_action.action
    prev_action = tf.nest.map_structure(lambda t: t[:-1], action)
    next_action = tf.nest.map_structure(lambda t: t[1:], action)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    loss = tf.add_n(tf.nest.flatten(distances))

    return loss, final_state, distances

  def sample(
      self,
      state_action: embed.StateActionReward,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[embed.Action, RecurrentState]:
    input = self.embed_state_action(state_action)
    output, final_state = self.network.step(input, initial_state)

    prev_action = state_action.action
    next_action = self.controller_head.sample(
        output, prev_action, **kwargs)
    return next_action, final_state
