from typing import Any, Sequence, Tuple
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai import data, networks
from slippi_ai.rl_lib import discounted_returns

RecurrentState = networks.RecurrentState
ControllerWithRepeat = dict

def get_p1_controller(game: data.CompressedGame) -> ControllerWithRepeat:
  return dict(
      controller=game.actions,
      action_repeat=game.counts)

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
    self.value_head = snt.Linear(1, name='value_head')

  def loss(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
      value_cost: float = 0.5,
      discount: float = 0.99,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    gamestates = compressed.states

    # compute policy loss
    p1_controller = get_p1_controller(compressed)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestates, p1_controller_embed)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    policy_loss = tf.add_n(tf.nest.flatten(distances))

    # compute value loss
    values = tf.squeeze(self.value_head(outputs), -1)
    num_frames = tf.cast(compressed.counts[1:] + 1, tf.float32)
    discounts = tf.pow(tf.cast(discount, tf.float32), num_frames)
    value_targets = discounted_returns(
        rewards=tf.cast(compressed.rewards[1:], tf.float32),
        discounts=discounts,
        bootstrap=values[-1])
    value_targets = tf.stop_gradient(value_targets)
    value_loss = tf.square(value_targets - values[:-1])
    value_stddev = tf.sqrt(tf.reduce_mean(value_loss))

    # compute metrics
    total_loss = policy_loss + value_cost * value_loss
    weighted_loss = tf.reduce_sum(total_loss) / tf.reduce_sum(num_frames)

    metrics = dict(
        policy=dict(
            distances,
            loss=policy_loss,
        ),
        value=dict(
            stddev=value_stddev,
            loss=value_loss,
        ),
        total_loss=total_loss,
        weighted_loss=weighted_loss,
    )

    return total_loss, final_state, metrics

  def sample(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
      **kwargs,
  ) -> Tuple[ControllerWithRepeat, RecurrentState]:
    gamestates = compressed.states
    p1_controller = get_p1_controller(compressed)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestates, p1_controller_embed)
    output, final_state = self.network.step(inputs, initial_state)

    controller_sample = self.controller_head.sample(
        output, p1_controller, **kwargs)
    return controller_sample, final_state
