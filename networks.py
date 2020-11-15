import sonnet as snt

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=dict(output_sizes=[256, 128]),
)

def mlp(output_sizes):
  return snt.nets.MLP(output_sizes, activate_final=True)

CONSTRUCTORS = dict(
    mlp=mlp,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
