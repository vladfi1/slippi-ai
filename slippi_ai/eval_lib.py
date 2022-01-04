import abc
import atexit
from dataclasses import dataclass
import functools
from typing import Callable, Dict, Mapping, NamedTuple, Tuple
import typing
import logging
from melee import console

import numpy as np
import tensorflow as tf

import melee

from slippi_ai import embed, policies, data, saving

expected_players = (1, 2)

class Policy(NamedTuple):
  sample: Callable[
      [data.CompressedGame, policies.RecurrentState],
      Tuple[policies.ControllerWithRepeat, policies.RecurrentState]]
  initial_state: Callable[[int], policies.RecurrentState]

  @staticmethod
  def from_saved_model(path: str) -> "Policy":
    policy = tf.saved_model.load(path)
    return Policy(
        sample=lambda *structs: policy.sample(*tf.nest.flatten(structs)),
        initial_state=policy.initial_state)

  @staticmethod
  def from_experiment(tag: str, sample_kwargs=None) -> "Policy":
    policy = saving.load_policy(tag)
    sample_kwargs = sample_kwargs or {}
    sample = functools.partial(policy.sample, **sample_kwargs)
    return Policy(
        sample=tf.function(sample),
        # sample=sample,
        initial_state=policy.initial_state)

class Agent:

  def __init__(
      self,
      controller: melee.Controller,
      opponent_port: int,
      policy: Policy,
  ):
    self._controller = controller
    self._port = controller.port
    self._players = (self._port, opponent_port)
    self._embed_game = embed.make_game_embedding(players=self._players)
    self._policy = policy
    self._sample = policy.sample
    self._hidden_state = policy.initial_state(1)
    self._current_action_repeat = 0
    self._current_repeats_left = 0

  def step(self, gamestate: melee.GameState):
    if self._current_repeats_left > 0:
      self._current_repeats_left -= 1
      return None

    embedded_game = self._embed_game.from_state(gamestate)
    # put the players in the expected positions
    embedded_game['player'] = {
      e: embedded_game['player'][p]
      for e, p in zip(expected_players, self._players)}

    unbatched_input = data.CompressedGame(embedded_game, self._current_action_repeat, 0.)
    batched_input = tf.nest.map_structure(
        lambda a: np.expand_dims(a, 0), unbatched_input)
    sampled_controller_with_repeat, self._hidden_state = self._sample(
        batched_input, self._hidden_state)
    sampled_controller_with_repeat = tf.nest.map_structure(
        lambda t: np.squeeze(t.numpy(), 0), sampled_controller_with_repeat)
    sampled_controller = sampled_controller_with_repeat['controller']
    self._current_action_repeat = sampled_controller_with_repeat['action_repeat']
    self._current_repeats_left = self._current_action_repeat

    for b in embed.LEGAL_BUTTONS:
      if sampled_controller['button'][b.value]:
        self._controller.press_button(b)
      else:
        self._controller.release_button(b)
    main_stick = sampled_controller["main_stick"]
    self._controller.tilt_analog(melee.Button.BUTTON_MAIN, *main_stick)
    c_stick = sampled_controller["c_stick"]
    self._controller.tilt_analog(melee.Button.BUTTON_C, *c_stick)
    self._controller.press_shoulder(melee.Button.BUTTON_L, sampled_controller["l_shoulder"])
    self._controller.press_shoulder(melee.Button.BUTTON_R, sampled_controller["r_shoulder"])

    return sampled_controller_with_repeat

class Player(abc.ABC):

  @abc.abstractmethod
  def controller_type(self) -> melee.ControllerType:
    pass

  @abc.abstractmethod
  def menuing_kwargs(self) -> Dict:
    pass


class Human(Player):

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.GCN_ADAPTER
  
  def menuing_kwargs(self) -> Dict:
      return {}

@dataclass
class CPU(Player):
  character: melee.Character = melee.Character.FOX
  level: int = 9

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character, cpu_level=self.level)

@dataclass
class AI(Player):
  character: melee.Character = melee.Character.FOX

  def controller_type(self) -> melee.ControllerType:
    return melee.ControllerType.STANDARD

  def menuing_kwargs(self) -> Dict:
      return dict(character_selected=self.character)

def _is_menu_state(gamestate: melee.GameState) -> bool:
  return gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]


class Dolphin:

  def __init__(
      self,
      dolphin_path: str,
      iso_path: str,
      players: Mapping[int, Player],
      stage: melee.Stage = melee.Stage.YOSHIS_STORY,
      online_delay=0,
      blocking_input=True,
  ) -> None:
    self._players = players
    self._stage = stage

    console = melee.Console(
        path=dolphin_path,
        online_delay=online_delay,
        blocking_input=blocking_input,
        copy_home_directory=False,
    )
    atexit.register(console.stop)
    self.console = console

    self.controllers = {}
    self._menuing_controllers = []
    self._autostart = True

    for port, player in players.items():
      controller = melee.Controller(
          console, port, player.controller_type())
      self.controllers[port] = controller
      if isinstance(player, Human):
        self._autostart = False
      else:
        self._menuing_controllers.append((controller, player))

    console.run(iso_path=iso_path)

    logging.info('Connecting to console...')
    if not console.connect():
      raise RuntimeError("Failed to connect to the console.")

    for controller in self.controllers.values():
      if not controller.connect():
        raise RuntimeError("Failed to connect the controller.")

  def next_gamestate(self) -> melee.GameState:
    gamestate = None
    while gamestate is None:
      gamestate = self.console.step()
    return gamestate

  def step(self) -> Tuple[melee.GameState, bool]:
    gamestate = self.next_gamestate()

    # The console object keeps track of how long your bot is taking to process frames
    #   And can warn you if it's taking too long
    if self.console.processingtime * 1000 > 12:
        print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

    menu_frames = 0
    while _is_menu_state(gamestate):
      for i, (controller, player) in enumerate(self._menuing_controllers):

        melee.MenuHelper.menu_helper_simple(
            gamestate, controller,
            stage_selected=self._stage,
            connect_code=None,
            autostart=self._autostart and i == 0 and menu_frames > 180,
            swag=False,
            costume=i,
            **player.menuing_kwargs())

      gamestate = self.next_gamestate()
      menu_frames += 1

    new_game = gamestate.frame == -123
    return gamestate, new_game

  def stop(self):
    self.console.stop()
