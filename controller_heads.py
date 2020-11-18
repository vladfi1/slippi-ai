import sonnet as snt
import tensorflow as tf

import embed

class ControllerHead(snt.Module):

  def sample(self, inputs, prev_controller_state):
    """Sample a controller state given input features and previous state."""

  def log_prob(self, inputs, prev_controller_state, target_controller_state):
    """A struct of log probabilities."""

class Independent(ControllerHead):
  """Models each component of the controller independently."""

  CONFIG = dict(
      residual=False,
  )

  def __init__(self, residual, embed_controller):
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

  def sample(self, inputs, prev_controller_state):
    return self.embed_controller.sample(
        self.controller_prediction(inputs, prev_controller_state))

  def log_prob(self, inputs, prev_controller_state, target_controller_state):
    return self.embed_controller.distance(
        self.controller_prediction(inputs, prev_controller_state),
        target_controller_state)

class AutoRegressive(ControllerHead):
  """Samples components sequentially conditioned on past samples."""

  CONFIG = dict(
  )

  def __init__(self, embed_controller):
    super().__init__(name='AutoRegressive')
    self.embed_controller = embed_controller
    self.embed_struct = self.embed_controller.map(lambda e: e)
    self.embed_flat = list(self.embed_controller.flatten(self.embed_struct))
    self.projections = [snt.Linear(e.size) for e in self.embed_flat]

  def sample(self, inputs, prev_controller_state):
    prev_controller_flat = self.embed_controller.flatten(prev_controller_state)

    samples = []
    for embedder, project, prev_component in zip(
        self.embed_flat, self.projections, prev_controller_flat):
      prev_component_embed = embedder(prev_component)

      # directly connect from the same component at time t-1
      input_ = tf.concat([inputs, prev_component_embed], -1)
      # project down to the size desired by the component
      input_ = project(input_)
      # sample the component
      sample = embedder.sample(input_)
      samples.append(sample)
      # auto-regress using the sampled component
      sample_repr = embedder(sample)
      # condition future components on the current sample
      inputs = tf.concat([inputs, sample_repr], -1)

    return self.embed_controller.unflatten(iter(samples))

  def log_prob(self, inputs, prev_controller_state, target_controller_state):
    prev_controller_flat = self.embed_controller.flatten(prev_controller_state)
    target_controller_flat = self.embed_controller.flatten(target_controller_state)
    logps = []

    for embedder, project, prev_component, target in zip(
        self.embed_flat, self.projections, prev_controller_flat,
        target_controller_flat):
      prev_component_embed = embedder(prev_component)

      # directly connect from the same component at time t-1
      input_ = tf.concat([inputs, prev_component_embed], -1)
      # project down to the size desired by the component
      input_ = project(input_)
      # compute the distance between prediction and target
      # in practice this is the cnegative log prob of the target
      logps.append(embedder.distance(input_, target))
      # auto-regress using the target (aka teacher forcing)
      sample_repr = embedder(target)
      # condition future components on the current sample
      inputs = tf.concat([inputs, sample_repr], -1)

    return tf.add_n(logps)

CONSTRUCTORS = dict(
    independent=Independent,
    autoregressive=AutoRegressive,
)

DEFAULT_CONFIG = dict({k: c.CONFIG for k, c in CONSTRUCTORS.items()})
DEFAULT_CONFIG.update(name='independent')

def construct(name, embed_controller, **config):
  kwargs = dict(config[name], embed_controller=embed_controller)
  return CONSTRUCTORS[name](**kwargs)
