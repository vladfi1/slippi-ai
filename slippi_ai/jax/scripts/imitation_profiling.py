"""Test imitation learning training loop - JAX version."""

from absl import app
import wandb
import fancyflags as ff

from slippi_ai import paths, flag_utils
from slippi_ai import data as data_lib
from slippi_ai.jax import train_lib

def network_config(num_layers: int, hidden_size: int) -> dict:
  return dict(
      name='tx_like',
      tx_like=dict(
          hidden_size=hidden_size,
          num_layers=num_layers,
          ffw_multiplier=2,
          recurrent_layer='lstm',
          activation='gelu',
      ),
  )

DEFAULT_CONFIG = train_lib.Config(
    dataset=data_lib.DatasetConfig(
        data_dir=str(paths.TOY_DATA_DIR),
        meta_path=str(paths.TOY_META_PATH),
        test_ratio=0.5,
    ),
    data=data_lib.DataConfig(
        batch_size=512,
        unroll_length=80,
        cached=True,
    ),
    runtime=train_lib.RuntimeConfig(
        log_interval=10,
        max_runtime=40,
        eval_every_n=10000,
    ),
    network=network_config(num_layers=3, hidden_size=512),
    value_function=train_lib.ValueFunctionConfig(
        separate_network_config=True,
        network=network_config(num_layers=1, hidden_size=512),
    ),
    controller_head=dict(
        name='autoregressive',
        autoregressive=dict(
            residual_size=128,
            component_depth=2,
        ),
    ),
)

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  CONFIG = ff.DEFINE_dict(
      'config', **flag_utils.get_flags_from_default(DEFAULT_CONFIG))


  def main(_):
    wandb.init(mode='disabled')

    config = flag_utils.dataclass_from_dict(
        train_lib.Config, CONFIG.value)

    train_lib.train(config)

  app.run(main)
