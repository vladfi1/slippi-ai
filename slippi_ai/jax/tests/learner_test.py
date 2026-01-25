"""Minimal test for Learner._step"""

import jax
import numpy as np
from flax import nnx

from slippi_ai import data as data_lib
from slippi_ai.jax import embed as embed_lib
from slippi_ai.jax import networks
from slippi_ai.jax import controller_heads
from slippi_ai.jax import policies
from slippi_ai.jax import learner as learner_lib
from slippi_ai.jax import value_function as vf_lib
from slippi_ai.jax import jax_utils


def make_dummy_frames(
    batch_size: int,
    unroll_length: int,
    embed_game: embed_lib.StructEmbedding[embed_lib.Game],
    embed_controller: embed_lib.StructEmbedding[embed_lib.Controller],
):
  """Create dummy frames for testing."""
  shape = (batch_size, unroll_length)

  dummy_state = embed_game.dummy(shape)
  dummy_action = embed_controller.dummy(shape)

  state_action = data_lib.StateAction(
      state=dummy_state,
      action=dummy_action,
      name=np.zeros(shape, dtype=np.int32),
  )

  frames = data_lib.Frames(
      state_action=state_action,
      is_resetting=np.zeros(shape, dtype=np.bool_),
      reward=np.zeros((batch_size, unroll_length - 1), dtype=np.float32),
  )

  return frames


def main():
  rngs = nnx.Rngs(0)

  # Config
  batch_size = 2
  unroll_length = 4
  num_names = 8
  hidden_size = 32

  # Create embeddings
  embed_config = embed_lib.EmbedConfig()
  embed_game = embed_config.make_game_embedding()
  embed_controller = embed_config.controller.make_embedding()

  # Create network
  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=embed_config,
      num_names=num_names,
      network_config=dict(
          name='mlp',
          mlp=dict(depth=2, width=hidden_size),
      ),
  )

  # Create controller head
  controller_head = controller_heads.construct(
      rngs=rngs,
      input_size=network.output_size,
      embed_controller=embed_controller,
      name='independent',
      independent={},
      autoregressive=controller_heads.AutoRegressive.default_config(),
  )

  # Create policy
  policy = policies.Policy(
      rngs=rngs,
      network=network,
      controller_head=controller_head,
      hidden_size=network.output_size,
      train_value_head=False,
      delay=0,
  )

  # Create a fake value function for testing
  value_function = vf_lib.FakeValueFunction()

  # Create learner
  learner = learner_lib.Learner(
      policy=policy,
      learning_rate=1e-4,
      value_cost=0.5,
      reward_halflife=4,
      value_function=value_function,
  )

  # Create dummy data
  frames = make_dummy_frames(batch_size, unroll_length, embed_game, embed_controller)

  # Get initial state
  initial_state = learner.initial_state(batch_size, rngs)

  print("Running _step (uncompiled)...")
  metrics, final_state = learner.step(frames, initial_state, train=True, compile=False)
  print(f"Total loss: {metrics['total_loss']}")

  print("Running step (JIT compiled, first call)...")
  metrics, final_state = learner.step(frames, initial_state, train=True, compile=True)
  print(f"Total loss: {metrics['total_loss']}")

  print("Running step (JIT compiled, second call)...")
  metrics, final_state = learner.step(frames, initial_state, train=True, compile=True)
  print(f"Total loss: {metrics['total_loss']}")

  # Multi-device test
  num_devices = jax.local_device_count()
  print(f"\nDetected {num_devices} device(s)")

  if num_devices > 1:
    print("\nTesting multi-device training...")

    # Create new learner for multi-device
    multi_device_learner = learner_lib.Learner(
        policy=policy,
        learning_rate=1e-4,
        value_cost=0.5,
        reward_halflife=4,
        value_function=value_function,
    )

    # Set up multi-device
    mesh = jax_utils.get_mesh()
    jax_utils.replicate_module(multi_device_learner, mesh)
    data_sharding = jax_utils.data_sharding(mesh)

    # Create data with batch size divisible by num_devices
    multi_batch_size = num_devices * 2
    multi_frames = make_dummy_frames(
        multi_batch_size, unroll_length, embed_game, embed_controller)
    multi_initial_state = multi_device_learner.initial_state(multi_batch_size, rngs)

    # Shard data
    multi_frames = jax_utils.shard_pytree(multi_frames, data_sharding)
    multi_initial_state = jax_utils.shard_pytree(multi_initial_state, data_sharding)

    print(f"Multi-device batch size: {multi_batch_size}")
    print(f"Per-device batch size: {multi_batch_size // num_devices}")

    print("Running multi-device step (first call)...")
    metrics, final_state = multi_device_learner.step(
        multi_frames, multi_initial_state, train=True)
    print(f"Total loss: {metrics['total_loss']}")

    print("Running multi-device step (second call)...")
    metrics, final_state = multi_device_learner.step(
        multi_frames, multi_initial_state, train=True)
    print(f"Total loss: {metrics['total_loss']}")

    print("Multi-device tests passed!")
  else:
    print("Skipping multi-device tests (only 1 device available)")

  print("\nSuccess!")


if __name__ == '__main__':
  main()
