import sonnet as snt
import tensorflow as tf

from slippi_ai import embed

class ControllerHead(snt.Module):

  def sample(self, inputs, prev_controller_state, temperature=None):
    """Sample a controller state given input features and previous state."""

  def distance(self, inputs, prev_controller_state, target_controller_state):
    """A struct of distances (generally, negative log probs)."""

class Independent(ControllerHead):
  """Models each component of the controller independently."""

  CONFIG = dict(
      residual=False,
  )

  def __init__(self, residual: bool, embed_controller: embed.Embedding):
    super().__init__(name='IndependentControllerHead')
    self.embed_controller = embed_controller
    self.to_controller_input = snt.Linear(self.embed_controller.size)
    self.residual = residual
    if residual:
      self.residual_net = snt.Linear(self.embed_controller.size,
        w_init=snt.initializers.Identity(), with_bias=False)

  def controller_prediction(self, inputs, prev_controller_state):
    controller_prediction = self.to_controller_input(inputs)
    if self.residual:
      prev_controller_flat = self.embed_controller(prev_controller_state)
      controller_prediction += self.residual_net(prev_controller_flat)
    return controller_prediction

  def sample(self, inputs, prev_controller_state, temperature=None):
    return self.embed_controller.sample(
        self.controller_prediction(inputs, prev_controller_state),
        temperature=temperature)

  def distance(self, inputs, prev_controller_state, target_controller_state):
    return self.embed_controller.distance(
        self.controller_prediction(inputs, prev_controller_state),
        target_controller_state)

class AutoRegressiveComponent(snt.Module):
  """Autoregressive residual component."""

  def __init__(self, embedder: embed.Embedding, residual_size, depth=0):
    super().__init__(name='ResBlock')
    self.embedder = embedder

    self.encoder = snt.nets.MLP([residual_size] * depth + [embedder.size])
    # the decoder doesn't need depth, because a single Linear decoding a one-hot
    # has full expressive power over the output
    self.decoder = snt.Linear(residual_size, w_init=tf.zeros_initializer())

  def sample(self, residual, prev_raw, **kwargs):
    # directly connect from the same component at time t-1
    prev_embedding = self.embedder(prev_raw)
    input_ = tf.concat([residual, prev_embedding], -1)
    # project down to the size desired by the component
    input_ = self.encoder(input_)
    # sample the component
    sample = self.embedder.sample(input_, **kwargs)
    # condition future components on the current sample
    sample_embedding = self.embedder(sample)
    residual += self.decoder(sample_embedding)
    return residual, sample

  def distance(self, residual, prev_raw, target_raw):
    # directly connect from the same component at time t-1
    prev_embedding = self.embedder(prev_raw)
    input_ = tf.concat([residual, prev_embedding], -1)
    # project down to the size desired by the component
    input_ = self.encoder(input_)
    # compute the distance between prediction and target
    distance = self.embedder.distance(input_, target_raw)
    # auto-regress using the target (aka teacher forcing)
    target_embedding = self.embedder(target_raw)
    residual += self.decoder(target_embedding)
    return residual, distance

class AutoRegressive(ControllerHead):
  """Samples components sequentially conditioned on past samples."""

  CONFIG = dict(
      residual_size=128,
      component_depth=0,
  )

  def __init__(
      self,
      embed_controller: embed.Embedding,
      residual_size: int,
      component_depth: int,
  ):
    super().__init__(name='AutoRegressive')
    self.embed_controller = embed_controller
    self.to_residual = snt.Linear(residual_size)
    self.embed_struct = self.embed_controller.map(lambda e: e)
    self.embed_flat = list(self.embed_controller.flatten(self.embed_struct))
    self.res_blocks = [
        AutoRegressiveComponent(e, residual_size, component_depth)
        for e in self.embed_flat]

  def sample(self, inputs, prev_controller_state, temperature=None):
    residual = self.to_residual(inputs)
    prev_controller_flat = self.embed_controller.flatten(prev_controller_state)

    samples = []
    for res_block, prev in zip(self.res_blocks, prev_controller_flat):
      residual, sample = res_block.sample(residual, prev, temperature=temperature)
      samples.append(sample)

    return self.embed_controller.unflatten(iter(samples))

  def distance(self, inputs, prev_controller_state, target_controller_state):
    residual = self.to_residual(inputs)
    prev_controller_flat = self.embed_controller.flatten(prev_controller_state)
    target_controller_flat = self.embed_controller.flatten(target_controller_state)

    distances = []
    for res_block, prev, target in zip(
        self.res_blocks, prev_controller_flat, target_controller_flat):
      residual, distance = res_block.distance(residual, prev, target)
      distances.append(distance)

    return self.embed_controller.unflatten(iter(distances))

CONSTRUCTORS = dict(
    independent=Independent,
    autoregressive=AutoRegressive,
)

DEFAULT_CONFIG = dict({k: c.CONFIG for k, c in CONSTRUCTORS.items()})
DEFAULT_CONFIG.update(name='independent')

def construct(name, embed_controller, **config):
  kwargs = dict(config[name], embed_controller=embed_controller)
  return CONSTRUCTORS[name](**kwargs)
