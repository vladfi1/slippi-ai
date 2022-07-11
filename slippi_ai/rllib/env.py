import functools
import typing as tp

from absl import logging
import gym
from gym import spaces
from ray import rllib
from ray.rllib.utils.typing import EnvType, AgentID, EnvID, MultiEnvDict

from melee.slippstream import EnetDisconnected
from slippi_ai import eval_lib, embed, reward
from slippi_ai import dolphin as dolphin_lib
from slippi_db import parse_libmelee

class AgentInfo(tp.NamedTuple):
  opponent_port: int
  embed_game: embed.Embedding = embed.default_embed_game
  embed_action: embed.Embedding = embed.embed_controller_discrete

class MeleeEnv(rllib.BaseEnv):

  def __init__(
      self,
      dolphin_fn: tp.Callable[[], dolphin_lib.Dolphin],
      num_envs: int = 1,
  ):
    self._dolphin_fn = dolphin_fn
    self._num_envs = num_envs

    self._dolphins = [dolphin_fn() for _ in range(num_envs)]

    players = self._dolphins[0]._players
    assert len(players) == 2

    self._agents: tp.Dict[int, AgentInfo] = {}
    ports = list(players)

    for port, opponent_port in zip(ports, reversed(ports)):
      player = players[port]
      if isinstance(player, dolphin_lib.AI):
        self._agents[port] = AgentInfo(opponent_port)

    self._prev_gamestates = None

  def _step_one(self, index: int, num_retries=3):
    """Steps dolphin, possibly creating a new instance if we disconnect."""
    for _ in range(num_retries):
      try:
        return self._dolphins[index].step()
      except EnetDisconnected:
        logging.warn('EnetDisconnected, restarting dolphin.')
        self._dolphins[index].stop()
        from ray.util import pdb; pdb.set_trace()
        self._dolphins[index] = self._dolphin_fn()
    return self._dolphins[index].step()

  def _step(self, num_retries=3):
    return [self._step_one(i) for i in range(self._num_envs)]

  def poll(
      self,
  ) -> tp.Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
    if self._prev_gamestates is None:
      self._prev_gamestates = self._step()

    all_envs_result = {}, {}, {}, {}, {}

    for env_id in range(self._num_envs):
      gamestate = self._step_one(env_id)

      observations = {}
      rewards = {}
      dones = {"__all__": False}
      infos = {}
      off_policy_actions = {}

      for port, info in self._agents.items():
        obs = parse_libmelee.get_game(
            gamestate, ports=(port, info.opponent_port))
        obs = info.embed_game.from_state(obs)
        obs = info.embed_game.to_nest(obs)
        observations[port] = obs

        rewards[port] = reward.get_reward(
            self._prev_gamestates[env_id], gamestate, port, info.opponent_port)
        dones[port] = False  # TODO: check if we just menued
        infos[port] = {}

      self._prev_gamestates[env_id] = gamestate

      result = observations, rewards, dones, infos, off_policy_actions
      for all_envs, value in zip(all_envs_result, result):
        all_envs[env_id] = value

    return all_envs_result

  def try_reset(self, env_id: EnvID = None) -> MultiEnvDict:
    # assert env_id in (None, _ENV_ID)
    return self.poll()[0]

  def send_actions(self, action_dict: dict):
    # print('send_actions')
    for env_id, agent_actions in action_dict.items():
      for port, controller in agent_actions.items():
        embed_action = self._agents[port].embed_action
        # convert from rllib Nests to our own NamedTuples
        controller = embed_action.from_nest(controller)
        # un-discretize sticks/shoulder
        controller = embed_action.decode(controller)
        eval_lib.send_controller(
            self._dolphins[env_id].controllers[port], controller)

  def get_agent_ids(self) -> tp.Set[AgentID]:
    return set(self._agents)

  def get_sub_environments(
      self, as_dict: bool = False,
  ) -> tp.Union[tp.List[EnvType], tp.Dict[EnvID, EnvType]]:
    if as_dict:
      return dict(enumerate(self._dolphins))
    else:
      return self._dolphins

  def stop(self):
    self._dolphin.stop()

  @functools.cached_property
  def observation_space(self) -> gym.Space:
    return spaces.Dict(
        {port: info.embed_game.space()
        for port, info in self._agents.items()
    })

  def observation_space_sample(self, agent_id: list = None) -> MultiEnvDict:
    if agent_id:
      sample = {a: self.observation_space[a].sample() for a in agent_id}
    else:
      sample = self.observation_space.sample()
    return {env_id: sample for env_id in range(self._num_envs)}

  @functools.cached_property
  def action_space(self) -> gym.Space:
    return spaces.Dict(
        {port: info.embed_action.space()
        for port, info in self._agents.items()
    })

  def action_space_sample(self, agent_id: list = None) -> MultiEnvDict:
    if agent_id:
      sample = {a: self.action_space[a].sample() for a in agent_id}
    else:
      sample = self.action_space.sample()
    return {env_id: sample for env_id in range(self._num_envs)}
