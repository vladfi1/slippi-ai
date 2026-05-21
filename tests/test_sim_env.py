import unittest

import melee
import numpy as np

from slippi_ai import dolphin
from slippi_ai import sim_env
from slippi_ai.sim_env import observations
from slippi_ai.types import Buttons, Controller, Items, Stick


class SimEnvTest(unittest.TestCase):

  def _sim_env(self, *args, **kwargs):
    try:
      return sim_env.SimBatchedEnvironment(*args, **kwargs)
    except MemoryError as exc:
      if 'msl_batch_create failed' not in str(exc):
        raise
      self.skipTest(
          'melee_sim EnvBatch could not initialize; set MELEE_SIM_DATA to an '
          'extracted melee-sim-light data directory before running sim env tests.')

  def test_current_state_and_step_match_existing_game_shape(self):
    env = self._sim_env(
        num_envs=3,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=8,
    )
    try:
      initial = env.current_state()
      self.assertEqual(set(initial.gamestates), {1, 2})
      self.assertEqual(initial.needs_reset.shape, (3,))
      self.assertTrue(np.all(initial.gamestates[1].p1.shield_strength == 60.0))
      self.assertEqual(
          initial.gamestates[1].p0.character.tolist(),
          [
              melee.Character.FOX.value,
              melee.Character.FOX.value,
              melee.Character.FOX.value,
          ],
      )
      self.assertEqual(
          initial.gamestates[1].p1.character.tolist(),
          [
              melee.Character.FALCO.value,
              melee.Character.FALCO.value,
              melee.Character.FALCO.value,
          ],
      )
      self.assertTrue(np.all(initial.gamestates[1].p0.jumps_left == 1))
      self.assertFalse(np.any(initial.gamestates[1].items.item_0.exists))
      self.assertEqual(
          initial.gamestates[2].p0.character.tolist(),
          [
              melee.Character.FALCO.value,
              melee.Character.FALCO.value,
              melee.Character.FALCO.value,
          ],
      )
      self.assertEqual(
          initial.gamestates[2].p1.character.tolist(),
          [
              melee.Character.FOX.value,
              melee.Character.FOX.value,
              melee.Character.FOX.value,
          ],
      )

      controllers = {
          1: sim_env.neutral_controllers(3),
          2: sim_env.neutral_controllers(3),
      }
      output = env.step(controllers)
      self.assertEqual(output.gamestates[1].p0.x.shape, (3,))
      self.assertTrue(np.all(output.gamestates[1].stage == melee.Stage.FINAL_DESTINATION.value))
      self.assertTrue(np.all(output.gamestates[1].p0.controller.main_stick.x == 0.5))
      self.assertTrue(np.all(output.gamestates[2].p0.controller.main_stick.x == 0.5))
      self.assertTrue(np.all(output.gamestates[1].p0.action >= 0))
      self.assertGreaterEqual(env.cursor, 1)
    finally:
      env.stop()

  def test_push_pop_queue_and_partial_reset(self):
    env = self._sim_env(num_envs=2, length=4)
    try:
      first = env.pop()
      self.assertTrue(np.all(first.needs_reset))

      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      env.push(controllers)
      stepped = env.pop()
      self.assertFalse(np.any(stepped.needs_reset))

      reset = env.reset([1])
      self.assertFalse(reset.needs_reset[0])
      self.assertTrue(reset.needs_reset[1])
      self.assertEqual(reset.gamestates[1].p0.x.shape, (2,))
    finally:
      env.stop()

  def test_partial_reset_clears_observed_previous_controllers(self):
    env = self._sim_env(num_envs=2, length=8)
    try:
      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      controllers[1].main_stick.x[:] = [0.0, 1.0]
      controllers[2].main_stick.x[:] = [0.25, 0.75]
      env.step(controllers)

      env.reset([1])
      state = env.current_game_batch(
          needs_reset=np.array([False, True], dtype=np.bool_))

      self.assertTrue(np.allclose(state.game.p0.controller.main_stick.x, [
          0.0,
          0.5,
          0.25,
          0.5,
      ]))
    finally:
      env.stop()

  def test_stage_assignment_cursor_wrap_and_controller_write(self):
    stages = [
        melee.Stage.FINAL_DESTINATION,
        melee.Stage.BATTLEFIELD,
        melee.Stage.YOSHIS_STORY,
    ]
    env = self._sim_env(num_envs=3, length=2, stage=stages)
    try:
      current = env.current_state()
      self.assertEqual(current.gamestates[1].stage.tolist(), [stage.value for stage in stages])

      controllers = {
          1: sim_env.neutral_controllers(3),
          2: sim_env.neutral_controllers(3),
      }
      controllers[1].main_stick.x[:] = [0.0, 0.25, 1.0]
      controllers[1].buttons.B[:] = [True, False, True]
      env.step(controllers)
      action = env.buffers.controller_action_view[0]['p'][:, 0]
      self.assertTrue(np.allclose(action['main_stick_x'], [0.0, 0.25, 1.0]))
      self.assertEqual(action['buttons']['B'].tolist(), [1, 0, 1])

      env.step(controllers)
      self.assertEqual(env.cursor, 2)
      env.step(controllers)
      self.assertEqual(env.cursor, 1)
    finally:
      env.stop()

  def test_per_env_character_pool(self):
    env = self._sim_env(
        num_envs=4,
        length=8,
        character_pool='fox,falco',
    )
    try:
      state = env.current_game_batch(
          needs_reset=np.ones(4, dtype=np.bool_))
      self.assertEqual(
          state.game.p0.character[:4].tolist(),
          [
              melee.Character.FOX.value,
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
              melee.Character.FALCO.value,
          ],
      )
      self.assertEqual(
          state.game.p0.character[4:].tolist(),
          [
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
              melee.Character.FOX.value,
              melee.Character.FALCO.value,
          ],
      )
    finally:
      env.stop()

  def test_character_pool_assignment_stays_fixed_on_reset(self):
    env = self._sim_env(
        num_envs=1,
        length=8,
        character_pool='fox,falco',
    )
    try:
      state = env.current_game_batch(np.ones(1, dtype=np.bool_))
      self.assertEqual(state.game.p0.character.tolist(), [
          melee.Character.FOX.value,
          melee.Character.FOX.value,
      ])

      state = env.reset([0]).gamestates[1]
      self.assertEqual(state.p0.character.tolist(), [melee.Character.FOX.value])
      self.assertEqual(state.p1.character.tolist(), [melee.Character.FOX.value])

      state = env.reset([0]).gamestates[1]
      self.assertEqual(state.p0.character.tolist(), [melee.Character.FOX.value])
      self.assertEqual(state.p1.character.tolist(), [melee.Character.FOX.value])
    finally:
      env.stop()

  def test_max_frame_terminal_is_reported_separately(self):
    env = self._sim_env(num_envs=2, length=128, max_frame_id=0)
    try:
      controllers = {
          1: sim_env.neutral_controllers(2),
          2: sim_env.neutral_controllers(2),
      }
      output = None
      for _ in range(123):
        output = env.step(controllers)
      self.assertIsNotNone(output)
      self.assertTrue(np.all(output.needs_reset))
      terminal = env.last_step_info.terminal
      self.assertTrue(np.all(terminal['done'] == 1))
      self.assertTrue(np.all(terminal['max_frame_reached'] == 1))
      self.assertTrue(np.all(terminal['match_ended'] == 0))
    finally:
      env.stop()

  def test_packed_state_and_encoded_step(self):
    env = self._sim_env(
        num_envs=2,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=8,
    )
    try:
      state = env.current_game_batch(
          needs_reset=np.ones(2, dtype=np.bool_))
      self.assertEqual(state.needs_reset.shape, (4,))
      self.assertEqual(state.game.p0.x.shape, (4,))
      self.assertEqual(state.game.p0.character[:2].tolist(), [
          melee.Character.FOX.value,
          melee.Character.FOX.value,
      ])
      self.assertEqual(state.game.p0.character[2:].tolist(), [
          melee.Character.FALCO.value,
          melee.Character.FALCO.value,
      ])
      self.assertEqual(state.game.p1.character[:2].tolist(), [
          melee.Character.FALCO.value,
          melee.Character.FALCO.value,
      ])
      self.assertEqual(state.game.p1.character[2:].tolist(), [
          melee.Character.FOX.value,
          melee.Character.FOX.value,
      ])

      encoded = _neutral_encoded_controller(batch_size=4)
      needs_reset = env.step_encoded(
          encoded,
          axis_spacing=32,
          shoulder_spacing=4,
      )
      self.assertEqual(needs_reset.shape, (2,))
      next_state = env.current_game_batch(needs_reset=needs_reset)
      self.assertEqual(next_state.game.p0.x.shape, (4,))
    finally:
      env.stop()

  def test_game_batch_matches_port_state_observation_conventions(self):
    env = self._sim_env(
        num_envs=3,
        players={
            1: dolphin.AI(melee.Character.FOX),
            2: dolphin.AI(melee.Character.FALCO),
        },
        length=8,
        stage=[
            melee.Stage.FINAL_DESTINATION,
            melee.Stage.BATTLEFIELD,
            melee.Stage.YOSHIS_STORY,
        ],
    )
    try:
      controllers = {
          1: sim_env.neutral_controllers(3),
          2: sim_env.neutral_controllers(3),
      }
      controllers[1].main_stick.x[:] = [0.0, 0.25, 1.0]
      controllers[1].buttons.A[:] = [True, False, True]
      controllers[2].main_stick.x[:] = [0.125, 0.5, 0.875]
      controllers[2].buttons.B[:] = [False, True, True]
      env.step(controllers)

      port_state = env.current_state()
      game_batch = env.current_game_batch(
          needs_reset=np.array([False, True, False], dtype=np.bool_))

      port1 = port_state.gamestates[1]
      port2 = port_state.gamestates[2]
      np.testing.assert_array_equal(game_batch.game.p0.action[:3], port1.p0.action)
      np.testing.assert_array_equal(game_batch.game.p1.action[:3], port1.p1.action)
      np.testing.assert_array_equal(game_batch.game.p0.action[3:], port2.p0.action)
      np.testing.assert_array_equal(game_batch.game.p1.action[3:], port2.p1.action)

      np.testing.assert_array_equal(game_batch.game.stage[:3], port1.stage)
      np.testing.assert_array_equal(game_batch.game.stage[3:], port2.stage)
      np.testing.assert_allclose(game_batch.game.randall.x[:3], port1.randall.x)
      np.testing.assert_allclose(game_batch.game.randall.x[3:], port2.randall.x)
      np.testing.assert_allclose(game_batch.game.fod_platforms.left, 0.0)
      np.testing.assert_allclose(game_batch.game.fod_platforms.right, 0.0)

      np.testing.assert_allclose(
          game_batch.game.p0.controller.main_stick.x,
          [0.0, 0.25, 1.0, 0.125, 0.5, 0.875],
      )
      np.testing.assert_allclose(
          game_batch.game.p1.controller.main_stick.x,
          [0.125, 0.5, 0.875, 0.0, 0.25, 1.0],
      )
      np.testing.assert_array_equal(
          game_batch.game.p0.controller.buttons.A,
          [True, False, True, False, False, False],
      )
      np.testing.assert_array_equal(
          game_batch.game.p1.controller.buttons.B,
          [False, True, True, False, False, False],
      )
      np.testing.assert_array_equal(
          game_batch.needs_reset,
          [False, True, False, False, True, False],
      )
    finally:
      env.stop()

  def test_slot_adapter_matches_libmelee_visible_conventions(self):
    slot = np.zeros(3, dtype=[
        ('present', np.bool_),
        ('stocks', np.uint8),
        ('percent', np.float32),
        ('facing', np.bool_),
        ('pos_x', np.float32),
        ('pos_y', np.float32),
        ('action_id', np.uint16),
        ('invulnerable', np.bool_),
        ('char_id', np.uint8),
        ('jumps_left', np.uint8),
        ('shield_hp', np.float32),
        ('on_ground', np.bool_),
    ])
    slot['present'] = True
    slot['stocks'] = 4
    slot['percent'] = [11.2, 11.8, 12.0]
    slot['char_id'] = melee.Character.FOX.value
    slot['jumps_left'] = [2, 2, 1]
    slot['on_ground'] = [True, False, False]
    slot['shield_hp'] = 60.0

    player = observations.player_from_slot(slot, sim_env.neutral_controllers(3))

    self.assertEqual(player.percent.tolist(), [11, 11, 12])
    self.assertEqual(player.jumps_left.tolist(), [2, 1, 1])

  def test_items_are_canonicalized_independent_of_backend_slot_order(self):
    items = np.zeros((1, len(Items._fields)), dtype=[
        ('exists', np.bool_),
        ('type', np.uint16),
        ('state', np.uint8),
        ('pos_x', np.float32),
        ('pos_y', np.float32),
    ])
    items[0, :4]['exists'] = [True, True, True, False]
    items[0, :4]['type'] = [74, 54, 74, 0]
    items[0, :4]['state'] = [4, 0, 5, 0]
    items[0, :4]['pos_x'] = [-54.0, -32.0, 60.0, 0.0]
    items[0, :4]['pos_y'] = [20.0, 40.0, 20.0, 0.0]

    out = observations.items_from_frame(items)

    self.assertEqual(out.item_0.type.tolist(), [74])
    self.assertEqual(out.item_0.x.tolist(), [-54.0])
    self.assertEqual(out.item_1.type.tolist(), [74])
    self.assertEqual(out.item_1.x.tolist(), [60.0])
    self.assertEqual(out.item_2.type.tolist(), [54])
    self.assertEqual(out.item_2.x.tolist(), [-32.0])
    self.assertFalse(out.item_3.exists[0])

def _neutral_encoded_controller(batch_size: int):
  shape = (int(batch_size),)
  return Controller(
      main_stick=Stick(
          x=np.full(shape, 16, dtype=np.uint8),
          y=np.full(shape, 16, dtype=np.uint8),
      ),
      c_stick=Stick(
          x=np.full(shape, 16, dtype=np.uint8),
          y=np.full(shape, 16, dtype=np.uint8),
      ),
      shoulder=np.zeros(shape, dtype=np.uint8),
      buttons=Buttons(**{
          name: np.zeros(shape, dtype=np.bool_)
          for name in Buttons._fields
      }),
  )


if __name__ == '__main__':
  unittest.main()
