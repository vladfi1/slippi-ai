from typing import Any, Sequence, Tuple
import sonnet as snt
import tensorflow as tf

from slippi_ai.controller_heads import ControllerHead
from slippi_ai import data, networks
from slippi_ai.rl_lib import discounted_returns

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
    self.value_head = snt.Linear(1, name='value_head')

  def loss(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
      value_cost: float = 0.5,
      discount: float = 0.99,
  ) -> Tuple[tf.Tensor, RecurrentState, dict]:
    gamestates = compressed.states
    action_repeat = compressed.counts
    num_frames = tf.cast(action_repeat[1:] + 1, tf.float32)

    # compute policy loss
    p1_controller = get_p1_controller(gamestates, action_repeat)

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

  def run(
      self,
      compressed: data.CompressedGame,
      initial_state: RecurrentState,
  ) -> Tuple[tf.Tensor, tf.Tensor, RecurrentState]:
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
    logprobs = -tf.add_n(tf.nest.flatten(distances))

    baseline = tf.squeeze(self.value_head(outputs), axis=-1)

    return logprobs, baseline, final_state

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
