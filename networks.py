import sonnet as snt

DEFAULT_CONFIG = dict(
    type='mlp',
    mlp=dict(output_sizes=[256, 128]),
)

def mlp(output_sizes):
  return snt.nets.MLP(output_sizes, activate_final=True)

CONSTRUCTORS = dict(
    mlp=mlp,
)

def construct_network(type, **config):
  return CONSTRUCTORS[type](**config[type])
