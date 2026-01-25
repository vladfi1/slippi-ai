"""Minimal multi-device training test with NNX."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from jax.sharding import Mesh, PartitionSpec, NamedSharding

P = PartitionSpec

class Model(nnx.Module):
  def __init__(self, rngs):
    self.linear = nnx.Linear(8, 4, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)

class Learner(nnx.Module):
  def __init__(self, model: Model):
    self.model = model
    self.optimizer = nnx.Optimizer(
        model, optax.adam(1e-4), wrt=nnx.Param)

  def train_step(self, x: jax.Array, y: jax.Array):
    def loss_fn(model):
      pred = model(x)
      return jnp.mean((pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(self.model)
    self.optimizer.update(self.model, grads)
    return loss

# Is there a cleaner way to do this in NNX?
@nnx.jit(donate_argnames=('learner',))
def train_step(learner: Learner, x: jax.Array, y: jax.Array):
  return learner.train_step(x, y)

def shard_module(
    model: nnx.Module,
    sharding: NamedSharding,
):
  """Replicates model parameters across devices."""
  _, state = nnx.split(model)
  sharded_state = jax.tree.map(
      lambda x: jax.device_put(x, sharding),
      state)
  nnx.update(model, sharded_state)

def main():
  num_devices = jax.local_device_count()
  print(f"Number of devices: {num_devices}")

  if num_devices < 2:
    print("Need at least 2 devices for this test")
    return

  # Create a mesh for data parallelism
  mesh = Mesh(jax.devices(), ('data',))

  # 1. Create a trivial model
  rngs = nnx.Rngs(0)
  model = Model(rngs)
  learner = Learner(model)

  # Replicate model
  shard_module(learner, NamedSharding(mesh, P()))

  # 2. Create fake data: [batch, features]
  batch_size = num_devices * 2  # 4 total, 2 per device
  x = jnp.ones((batch_size, 8))
  y = jnp.zeros((batch_size, 4))

  print(f"Input shape: {x.shape}")
  print(f"Target shape: {y.shape}")

  # 3. Shard data across devices: [num_devices, per_device_batch, features]
  def shard_batch(arr: jax.Array) -> jax.Array:
    sharding = NamedSharding(mesh, P('data'))
    return jax.device_put(arr, sharding)

  x_sharded = shard_batch(x)
  y_sharded = shard_batch(y)

  jax.debug.visualize_array_sharding(x_sharded)
  jax.debug.visualize_array_sharding(y_sharded)

  # 7. Run training step
  print("\nRunning training step...")
  loss = train_step(learner, x_sharded, y_sharded)

  print(f"Loss shape: {loss.shape}")
  print(f"Loss values: {loss}")

  # 9. Verify it works for multiple steps
  print("\nRunning 3 more steps...")
  for i in range(3):
    loss = train_step(learner, x_sharded, y_sharded)
    print(f"  Step {i+1}: loss = {loss:.6f}")

  print("\nSuccess!")


if __name__ == '__main__':
  main()
