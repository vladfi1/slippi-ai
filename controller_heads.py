import sonnet as snt
import tensorflow as tf

import embed

class ControllerHead(snt.Module):

  def sample(self, inputs, prev_controller_state):
    """Sample a controller state given input features and previous state."""

  def log_prob(self, inputs, prev_controller_state, target_controller_state):
    """A struct of log probabilities."""

class IndependentControllerHead(ControllerHead):
  """Models each component of the controller independently."""

  CONFIG = dict(
      residual=False,
      discrete_axis=False,
  )

  def __init__(self, residual, discrete_axis):
    super().__init__(name='IndependentControllerHead')
    self.embed_controller = embed.embed_controller
    if discrete_axis:
        self.embed_controller = embed.embed_controller_discrete
    self.to_controller_input = snt.Linear(self.embed_controller.size)
    self.residual = residual
    if residual:
      self.residual_net = snt.Linear(self.embed_controller.size)

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

CONSTRUCTORS = dict(
    independent=IndependentControllerHead,
)

DEFAULT_CONFIG = dict({k: c.CONFIG for k, c in CONSTRUCTORS.items()})
DEFAULT_CONFIG.update(name='independent')

def construct(name, **config):
  return CONSTRUCTORS[name](**config[name])
