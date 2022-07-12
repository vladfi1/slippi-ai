import typing as tp

import tree
import numpy as np
import tensorflow as tf
import sonnet as snt
import gym

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.tf_action_dist import ActionDistribution
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from slippi_ai import networks, controller_heads, embed

class SlippiModel(TFModelV2):

  def __init__(
      self,
      obs_space: gym.spaces.Space,
      action_space: gym.spaces.Space,
      num_outputs: int,
      model_config: ModelConfigDict,
      name: str,
      network: dict,
      controller_head: dict,
  ):
    super().__init__(
        obs_space, action_space, num_outputs, model_config, name
    )
    self.time_major = True

    self.embed_game = embed.default_embed_game
    # assert self.embed_game.space() == obs_space
    self.obs_space = self.embed_game.space()

    self.embed_controller = embed.embed_controller_discrete
    # assert self.embed_controller.space() == action_space

    assert model_config["_disable_preprocessor_api"]
    self.network = networks.construct_network(**network)

    controller_head_config = dict(
        controller_head,
        embed_controller=self.embed_controller)
    self.controller_head = controller_heads.construct(**controller_head_config)

    self.embed_state_action = embed.get_state_action_embedding(
        self.embed_game, self.embed_controller)

    self.value_head = snt.Sequential([
        snt.Linear(1), lambda x: tf.squeeze(x, -1)])

    self.modules = [
        self.network,
        self.controller_head,
        self.value_head,
    ]

    self._initial_state = self.network.initial_state(1)
    self._init_vars()

  def _init_vars(self):
    initial_state = self.network.initial_state(1)
    obs = self.embed_game.dummy([1])
    self.forward({'obs': obs}, tf.nest.flatten(initial_state))

  def variables(self, as_dict: bool = False) -> tp.Union[
      tp.List[TensorType], tp.Dict[str, TensorType]]:
    vars = []
    for module in self.modules:
      vars.extend(module.variables)
    if as_dict:
      return {v.name: v for v in vars}
    return vars

  def get_initial_state(self) -> tp.List[np.ndarray]:
    batched_state = self.network.initial_state(1)
    return tree.map_structure(
        lambda t: t[0].numpy(),
        batched_state
    )

  def forward(
      self,
      input_dict: tp.Dict[str, TensorType],
      state: tp.List[TensorType],
      seq_lens: TensorType,
  ) -> tp.Tuple[TensorType, tp.List[TensorType]]:
    obs = self.embed_game.from_nest(input_dict['obs'])

    prev_action = self.embed_controller.from_nest(
        input_dict['prev_action'])
    self.prev_action = prev_action

    sar = embed.StateActionReward(
      obs, prev_action, input_dict['prev_reward'])
    inputs: tf.Tensor = self.embed_state_action(sar)
    # inputs: tf.Tensor = self.embed_game(obs)

    state = tf.nest.pack_sequence_as(self._initial_state, state)

    assert len(inputs.shape) == 2
    outputs, next_state = self.network.step(inputs, state)
    self._value_out = self.value_head(outputs)

    next_state = tf.nest.flatten(next_state)
    return outputs, next_state

  def value_function(self) -> TensorType:
    return self._value_out

class SlippiActionDist(ActionDistribution):
  
  def sample(self) -> TensorType:
    model: SlippiModel = self.model

    model.controller_head.sample(
      inputs=self.inputs,
      prev_controller_state=)

def register():
  ModelCatalog.register_custom_model('slippi', SlippiModel)
