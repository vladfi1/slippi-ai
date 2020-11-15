import sonnet as snt

DEFAULT_CONFIG = dict(
    name='mlp',
    mlp=dict(
      output_sizes=[256, 128],
      dropout_rate=0.,
    ),
)

def mlp(output_sizes, dropout_rate):
  return snt.nets.MLP(output_sizes, activate_final=True,
    dropout_rate=dropout_rate)

CONSTRUCTORS = dict(
    mlp=mlp,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
