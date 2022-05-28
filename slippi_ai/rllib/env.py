import functools
import typing as tp

import gym
from gym import spaces
from ray import rllib
from ray.rllib.utils.typing import EnvType, AgentID, EnvID, MultiEnvDict

from slippi_ai import eval_lib, embed, reward
from slippi_ai import dolphin as dolphin_lib
from slippi_db import parse_libmelee

_ENV_ID = 0

class AgentInfo(tp.NamedTuple):
  opponent_port: int
  embed_game: embed.Embedding = embed.default_embed_game
  embed_action: embed.Embedding = embed.embed_controller_discrete

class MeleeEnv(rllib.BaseEnv):

  def __init__(
      self,
      dolphin: dolphin_lib.Dolphin,
  ):
    self._dolphin = dolphin

    players = dolphin._players
    assert len(players) == 2

    self._agents: tp.Dict[int, AgentInfo] = {}
    ports = list(players)

    for port, opponent_port in zip(ports, reversed(ports)):
      player = players[port]
      if isinstance(player, dolphin_lib.AI):
        self._agents[port] = AgentInfo(opponent_port)

    self._prev_gamestate = None

  def poll(
      self,
  ) -> tp.Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
    if self._prev_gamestate is None:
      self._prev_gamestate = self._dolphin.step()
    gamestate = self._dolphin.step()

    observations = {}
    rewards = {}
    dones = {"__all__": False}
    infos = {}
    off_policy_actions = {}

    for port, info in self._agents.items():
      obs = parse_libmelee.get_game(gamestate, ports=(port, info.opponent_port))
      obs = info.embed_game.from_state(obs)
      obs = info.embed_game.to_nest(obs)
      observations[port] = obs

      rewards[port] = reward.get_reward(
          self._prev_gamestate, gamestate, port, info.opponent_port)
      dones[port] = False  # TODO: check if we just menued
      infos[port] = {}

    self._prev_gamestate = gamestate

    result = observations, rewards, dones, infos, off_policy_actions
    result = tuple({_ENV_ID: x} for x in result)
    return result

  def try_reset(self, env_id: EnvID = None) -> MultiEnvDict:
    obs = self.poll()[0]
    return obs if env_id is None else obs[env_id]

  def send_actions(self, action_dict: dict):
    # print('send_actions')
    agent_actions = action_dict[_ENV_ID]
    for port, controller in agent_actions.items():
      embed_action = self._agents[port].embed_action
      # convert from rllib Nests to our own NamedTuples
      controller = embed_action.from_nest(controller)
      # un-discretize sticks/shoulder
      controller = embed_action.decode(controller)
      eval_lib.send_controller(
          self._dolphin.controllers[port], controller)

  def get_agent_ids(self) -> tp.Set[AgentID]:
    return set(self._agents)

  def get_sub_environments(
      self, as_dict: bool = False,
  ) -> tp.Union[tp.List[EnvType], tp.Dict[EnvID, EnvType]]:
    if as_dict:
      return {_ENV_ID: self}
    else:
      return [self]

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
    return {_ENV_ID: sample}

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
    return {_ENV_ID: sample}
