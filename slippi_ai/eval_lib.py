import functools
import multiprocessing as mp
from typing import Callable, Mapping, NamedTuple, Tuple

import numpy as np
import tensorflow as tf

import melee

from slippi_ai import embed, policies, data, saving, dolphin

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

def send_controller(controller: melee.Controller, controller_state: dict):
  for b in embed.LEGAL_BUTTONS:
    if controller_state['button'][b.value]:
      controller.press_button(b)
    else:
      controller.release_button(b)
  main_stick = controller_state["main_stick"]
  controller.tilt_analog(melee.Button.BUTTON_MAIN, *main_stick)
  c_stick = controller_state["c_stick"]
  controller.tilt_analog(melee.Button.BUTTON_C, *c_stick)
  controller.press_shoulder(melee.Button.BUTTON_L, controller_state["l_shoulder"])
  controller.press_shoulder(melee.Button.BUTTON_R, controller_state["r_shoulder"])

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
    self._embed_game = embed.make_game_embedding(ports=self._players)
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
    # embedded_game['player'] = {
    #   e: embedded_game['player'][p]
    #   for e, p in zip(expected_players, self._players)}

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

    send_controller(self._controller, sampled_controller)

    return sampled_controller_with_repeat


class Environment(dolphin.Dolphin):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    assert len(self._players) == 2

    self._game_embedders = {}
    ports = list(self._players)

    for port, opponent_port in zip(ports, reversed(ports)):
      if isinstance(self._players[port], dolphin.AI):
        self._game_embedders[port] = embed.make_game_embedding(
            ports=(port, opponent_port))

  def step(self, controllers: Mapping[int, dict]):
    for port, controller in controllers.items():
      send_controller(self.controllers[port], controller)
    
    gamestate = super().step()

    return {port: e.from_state(gamestate) for port, e in self._game_embedders.items()}


def run_env(init_kwargs, conn):
  env = Environment(**init_kwargs)

  while True:
    controllers = conn.recv()
    if controllers is None:
      break

    conn.send(env.step(controllers))

  env.stop()
  conn.close()

class AsyncEnv:

  def __init__(self, **kwargs):
    self._parent_conn, child_conn = mp.Pipe()
    self._process = mp.Process(target=run_env, args=(kwargs, child_conn))
    self._process.start()

  def stop(self):
    self._parent_conn.send(None)
    self._process.join()
    self._parent_conn.close()

  def send(self, controllers):
    self._parent_conn.send(controllers)
  
  def recv(self):
    return self._parent_conn.recv()