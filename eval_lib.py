from typing import Optional

import numpy as np
import tensorflow as tf

import melee
import embed

class SavedModelPolicy:

  def __init__(
      self,
      controller: melee.Controller,
      opponent_port: int,
      policy = None,
      path: Optional[str] = None,
  ):
    self._controller = controller
    self._port = controller.port
    self._opponent_port = opponent_port
    self._embed_game = embed.make_game_embedding(players=(self._port, opponent_port))
    self._policy = policy or tf.saved_model.load(path)
    self._sample = lambda *structs: self._policy.sample(*tf.nest.flatten(structs))
    self._hidden_state = self._policy.initial_state(1)
    self._current_action_repeat = 0
    self._current_repeats_left = 0

  def step(self, gamestate: melee.GameState):
    if self._current_repeats_left > 0:
      self._current_repeats_left -= 1
      return

    embedded_game = self._embed_game.from_state(gamestate), self._current_action_repeat, 0.
    batched_game = tf.nest.map_structure(
        lambda a: np.expand_dims(a, 0), embedded_game)
    sampled_controller_with_repeat, self._hidden_state = self._sample(
        batched_game, self._hidden_state)
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
