"""Tests that exported ONNX agents match their JAX policies.

Exports each checkpoint, runs the OnnxAgent on replay frames, and replays the
same sampling noise through the JAX step function, checking that the sampled
(decoded) controllers and recurrent states match.

Requires the onnx-export extra: pip install .[onnx-export]
"""

import os

from absl import app, flags
import jax
import numpy as np

from slippi_ai import data, paths, saving, utils
from slippi_ai.jax import onnx_export
from slippi_ai import onnx_policies

flags.DEFINE_list(
    'models', ['jax_demo', 'jax_merged_demo', 'jax_policy_demo'],
    f'Checkpoints in {paths.CHECKPOINTS_DIR} to test.')
flags.DEFINE_integer('steps', 60, 'Number of frames to run.')
flags.DEFINE_integer('reset_every', 25, 'Reset the agent every this many frames.')

FLAGS = flags.FLAGS

BATCH_SIZE = 2


class RecordingRng:
  """Wraps a numpy Generator, recording the noise drawn by the agent."""

  def __init__(self, seed: int):
    self._rng = np.random.default_rng(seed)
    self.draws: list[np.ndarray] = []

  def random(self, shape, dtype):
    x = self._rng.random(shape, dtype=dtype)
    self.draws.append(x)
    return x


def frames(replay, t: int):
  """Batch of two replay frames, offset so that they differ."""
  return utils.map_nt(lambda x: x[[t, t + 1000]], replay)


def load_replay():
  path = os.path.join(paths.TOY_DATA_DIR, sorted(os.listdir(paths.TOY_DATA_DIR))[0])
  return data.read_table(path, compressed=True)


def test_model(model: str, replay):
  state = saving.load_state_from_disk(str(paths.CHECKPOINTS_DIR / model))
  onnx_model = onnx_export.export_state(state).SerializeToString()

  onnx_policy = onnx_policies.OnnxPolicy(
      onnx_model, providers=['CPUExecutionProvider'])
  agent = onnx_policy.build_agent(
      BATCH_SIZE, name_code=0, rating=1500, sample_kwargs=dict(temperature=0.8))
  rng = RecordingRng(seed=0)
  agent._rng = rng  # pylint: disable=protected-access

  # Run the ONNX agent, keeping its outputs for every frame.
  frame_skip = onnx_policy.frame_skip
  onnx_controllers = []
  onnx_states = []
  for t in range(FLAGS.steps):
    game = frames(replay, t)
    needs_reset = np.array([t % FLAGS.reset_every == 0] * BATCH_SIZE)
    onnx_controllers.append(agent.step(game, needs_reset).controller_state)
    if t % frame_skip == frame_skip - 1:
      onnx_states.append(agent.hidden_state())

  # Replay the same noise through the JAX step function.
  policy = saving.load_policy_from_state(state)
  game_embedding = onnx_export.game_embedding_from_config(state['config'])
  step = jax.jit(onnx_export.make_step_fn(policy, game_embedding))
  num_noise = len(onnx_export.noise_specs(policy, game_embedding))

  inputs = onnx_export.dummy_step_inputs(policy, BATCH_SIZE)._replace(
      name=np.zeros([BATCH_SIZE], np.int32),
      rating=np.full([BATCH_SIZE], 1500, np.float32),
      temperature=np.array(0.8, np.float32),
  )
  jax_controllers = []
  jax_states = []
  needs_reset = np.full([BATCH_SIZE], False)
  for i, t in enumerate(range(0, FLAGS.steps, frame_skip)):
    # Resets during skipped frames apply at the next policy call.
    for s in range(t - frame_skip + 1, t + 1):
      if s >= 0 and s % FLAGS.reset_every == 0:
        needs_reset[:] = True
    game = frames(replay, t)
    noise = rng.draws[i * num_noise:(i + 1) * num_noise]
    outputs = step(inputs._replace(
        game=game, needs_reset=needs_reset.copy(), noise=noise))
    needs_reset[:] = False
    inputs = inputs._replace(prev_actions=outputs.actions, prev_state=outputs.state)
    # Checks the OnnxAgent's numpy decoding against the policy's.
    jax_controllers.extend(
        policy.controller_head.decode_controller(jax.tree.map(np.asarray, a))
        for a in outputs.actions)
    jax_states.append(outputs.state)

  # Compare.
  for t, (c_onnx, c_jax) in enumerate(zip(onnx_controllers, jax_controllers)):
    for (path, x), y in zip(
        jax.tree_util.tree_flatten_with_path(c_jax)[0], jax.tree.leaves(c_onnx)):
      if not np.array_equal(np.asarray(x), y):
        raise AssertionError(
            f'{model}: controller {jax.tree_util.keystr(path)} differs at frame {t}:'
            f' jax={np.asarray(x)} onnx={y}')

  for i, (s_onnx, s_jax) in enumerate(zip(onnx_states, jax_states)):
    names, leaves, _ = onnx_export._flatten_with_names(  # pylint: disable=protected-access
        onnx_export.StepInputs(*[None] * 6, prev_state=s_jax, noise=None))
    for name, x in zip(names, leaves):
      np.testing.assert_allclose(
          s_onnx[name], np.asarray(x), rtol=1e-4, atol=1e-5,
          err_msg=f'{model}: state {name} differs at policy step {i}')

  print(f'{model}: {FLAGS.steps} frames match (frame_skip={frame_skip})')


def main(_):
  replay = load_replay()
  for model in FLAGS.models:
    test_model(model, replay)


if __name__ == '__main__':
  app.run(main)
