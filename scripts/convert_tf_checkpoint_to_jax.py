"""Convert legacy TensorFlow policy checkpoints into native JAX checkpoints."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import pickle
from pathlib import Path
import typing as tp

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import tree

from slippi_ai import data, observations, paths, reward, utils
from slippi_ai.flag_utils import dataclass_from_dict
from slippi_ai.jax import controller_heads, embed, jax_utils, networks, policies, saving


ArrayTree = dict[str | int, tp.Any]
_VERIFY_MAX_GAMES = 5
_VERIFY_SUBSAMPLE = 100


def _path_parts(path: str) -> list[str | int]:
  result: list[str | int] = []
  for part in path.split('/'):
    result.append(int(part) if part.isdecimal() else part)
  return result


def _get_path(tree_: ArrayTree, path: str):
  cursor = tree_
  for part in _path_parts(path):
    cursor = cursor[part]
  return cursor


def _set_path(params: ArrayTree, path: str, value: np.ndarray) -> None:
  cursor = params
  parts = _path_parts(path)
  for part in parts[:-1]:
    cursor = cursor[part]
  expected = np.asarray(cursor[parts[-1]])
  value = np.asarray(value)
  if value.shape != expected.shape:
    raise ValueError(
        f'shape mismatch for {path}: source {value.shape} != target {expected.shape}')
  cursor[parts[-1]] = value


def _numeric_keys(mapping: dict) -> list[int]:
  return sorted(key for key in mapping if isinstance(key, int))


def _split_lstm_gates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  return tuple(np.split(np.asarray(x), 4, axis=-1))  # type: ignore[return-value]


def _combined_kernel_shape(core: dict, gates: tuple[str, ...]) -> tuple[int, int]:
  kernels = [np.asarray(core[gate]['kernel']) for gate in gates]
  rows = {kernel.shape[0] for kernel in kernels}
  if len(rows) != 1:
    raise ValueError(f'inconsistent LSTM gate kernel input sizes: {rows}')
  return (rows.pop(), sum(kernel.shape[1] for kernel in kernels))


class LeafReader:
  """Sequential TF leaf reader with target-shape validation."""

  def __init__(self, source_state):
    self._leaves = [np.asarray(x) for x in tree.flatten(source_state)]
    self._index = 0

  @property
  def remaining(self) -> int:
    return len(self._leaves) - self._index

  def read(self, expected_shape: tuple[int, ...], label: str) -> np.ndarray:
    if self._index >= len(self._leaves):
      raise ValueError(f'ran out of TensorFlow leaves while reading {label}')
    value = self._leaves[self._index]
    if value.shape != expected_shape:
      raise ValueError(
          f'shape mismatch for TF leaf {self._index} ({label}): '
          f'{value.shape} != {expected_shape}')
    self._index += 1
    return value

  def finish(self):
    if self._index != len(self._leaves):
      raise ValueError(
          f'converted {self._index} TensorFlow leaves but checkpoint has '
          f'{len(self._leaves)} leaves')


def _jax_config_from_tf_config(config: dict) -> dict:
  config = copy.deepcopy(config)

  # Old TF checkpoints predate the current JAX observation/embed config shape.
  # Normalize the saved config first so the target JAX Policy is constructed
  # with exactly the modules whose parameters we are about to populate.
  if config.get('version') == 3:
    config['observation'] = dataclasses.asdict(
        observations.NULL_OBSERVATION_CONFIG)
    config['version'] = 4

  if config.get('version') == 4:
    config['embed'].update(
        with_randall=False,
        with_fod=False,
        items={'type': embed.ItemsType.SKIP},
    )
    config['embed']['player'].update(
        with_nana=False,
        legacy_jumps_left=True,
    )
    config['version'] = 5

  if config.get('version') != 5:
    raise ValueError(f"Unsupported legacy TF checkpoint version: {config.get('version')}")

  controller = config['embed']['controller']
  config['embed']['controller'] = {
      'type': 'default',
      'default': controller,
  }

  _upgrade_network_config(config['network'])
  _upgrade_network_config(config['value_function']['network'])
  config.setdefault('seed', 0)
  config['platform'] = 'jax'
  config['version'] = saving.VERSION
  return config


def _upgrade_network_config(config: dict):
  config.setdefault('embed', {
      'name': 'simple',
      'simple': {},
      'enhanced': networks.EnhancedEmbedModule.default_config(),
  })
  if config['name'] == 'tx_like':
    config['tx_like']['layer_norm'] = 'tensorflow'
    config['tx_like']['gelu_approximate'] = False


def _policy_from_config(config: dict) -> policies.Policy:
  embed_config = dataclass_from_dict(embed.EmbedConfig, config['embed'])
  policy_config = dataclass_from_dict(policies.PolicyConfig, config['policy'])
  rngs = nnx.Rngs(config.get('seed', 0))

  network = networks.build_embed_network(
      rngs=rngs,
      embed_config=embed_config,
      num_names=config['max_names'],
      network_config=config['network'],
  )
  controller_head = controller_heads.construct(
      rngs=rngs,
      input_size=network.output_size,
      embed_controller=embed_config.controller.make_embedding(),
      **config['controller_head'],
  )
  return policies.Policy(
      network=network,
      controller_head=controller_head,
      delay=policy_config.delay,
  )


def _convert_autoregressive_head(params: ArrayTree, reader: LeafReader) -> None:
  head = params['_controller_head']
  if 'res_blocks' not in head or 'to_residual' not in head:
    raise ValueError('only legacy autoregressive TensorFlow controller heads are supported')

  # The legacy TF pickle stores leaves in module traversal order. The JAX head
  # has the same logical layers but different tree names, so conversion is a
  # shape-checked replay of that known order into the target parameter tree.
  for block_index in _numeric_keys(head['res_blocks']):
    block = head['res_blocks'][block_index]
    base = f'_controller_head/res_blocks/{block_index}'
    _set_from_leaf(params, reader, f'{base}/decoder/bias')
    _set_from_leaf(params, reader, f'{base}/decoder/kernel')
    for layer_index in _numeric_keys(block['_encoder']['layers']):
      layer_base = f'{base}/_encoder/layers/{layer_index}'
      _set_from_leaf(params, reader, f'{layer_base}/bias')
      _set_from_leaf(params, reader, f'{layer_base}/kernel')

  _set_from_leaf(params, reader, '_controller_head/to_residual/bias')
  _set_from_leaf(params, reader, '_controller_head/to_residual/kernel')


def _set_from_leaf(params: ArrayTree, reader: LeafReader, path: str) -> None:
  expected_shape = np.asarray(_get_path(params, path)).shape
  value = reader.read(expected_shape, path)
  _set_path(params, path, value)


def _convert_network(params: ArrayTree, reader: LeafReader, network_config: dict) -> None:
  if network_config['name'] != 'tx_like':
    raise ValueError(
        f"only legacy tx_like TensorFlow policy networks are supported; "
        f"got {network_config['name']!r}")

  layers = params['network']['_network']['_layers']
  layer_indices = _numeric_keys(layers)
  if not layer_indices or layer_indices[0] != 0:
    raise ValueError('expected tx_like network layer 0 to be the input encoder')

  _set_from_leaf(params, reader, 'network/_network/_layers/0/_module/bias')
  _set_from_leaf(params, reader, 'network/_network/_layers/0/_module/kernel')

  remaining = layer_indices[1:]
  if len(remaining) % 2:
    raise ValueError(f'expected recurrent/FFW layer pairs after encoder, got {remaining}')

  for recurrent_layer, ffw_layer in zip(remaining[::2], remaining[1::2]):
    _convert_lstm_layer(params, reader, recurrent_layer)
    ffw_base = f'network/_network/_layers/{ffw_layer}/_module'
    _set_from_leaf(params, reader, f'{ffw_base}/layernorm/bias')
    _set_from_leaf(params, reader, f'{ffw_base}/layernorm/scale')
    _set_from_leaf(params, reader, f'{ffw_base}/linear1/bias')
    _set_from_leaf(params, reader, f'{ffw_base}/linear1/kernel')
    _set_from_leaf(params, reader, f'{ffw_base}/linear2/bias')
    _set_from_leaf(params, reader, f'{ffw_base}/linear2/kernel')


def _skip_legacy_policy_value_readout(reader: LeafReader, network_config: dict) -> None:
  """Skip the TF policy tuple's trailing value readout, if present.

  Legacy TF policy checkpoints can carry a scalar value readout at the end of
  the policy parameter tuple even though the saved checkpoint also has a
  separate `state["value_function"]`. JAX `Policy` has no corresponding module,
  and the previous converter did not use these leaves. Keep that behavior
  explicit and shape-checked instead of silently accepting arbitrary leftovers.
  """
  if reader.remaining == 0:
    return

  tx_like = network_config.get('tx_like', {})
  hidden_size = int(tx_like.get('hidden_size', 0))
  if reader.remaining != 2 or hidden_size <= 0:
    raise ValueError(
        f'unsupported extra TensorFlow policy leaves: {reader.remaining} remain')

  reader.read((1,), 'legacy policy value readout bias')
  reader.read((hidden_size, 1), 'legacy policy value readout kernel')


def _convert_lstm_layer(params: ArrayTree, reader: LeafReader, layer_index: int) -> None:
  core_base = f'network/_network/_layers/{layer_index}/_net/_core'
  core = _get_path(params, core_base)
  if not all(gate in core for gate in ('hi', 'hf', 'hg', 'ho', 'ii', 'if_', 'ig', 'io')):
    raise ValueError(f'only LSTM tx_like recurrent layers are supported: {core_base}')

  hidden_shape = _combined_kernel_shape(core, ('hi', 'hf', 'hg', 'ho'))
  input_shape = _combined_kernel_shape(core, ('ii', 'if_', 'ig', 'io'))
  bias_shape = (sum(np.asarray(core[gate]['bias']).shape[0]
                    for gate in ('hi', 'hf', 'hg', 'ho')),)

  hidden_kernel = reader.read(hidden_shape, f'{core_base}/hidden_kernel')
  input_kernel = reader.read(input_shape, f'{core_base}/input_kernel')
  bias = reader.read(bias_shape, f'{core_base}/bias')

  # TF keeps each LSTM core as combined i/f/g/o matrices; the JAX module stores
  # one small linear per gate. Split the combined tensors without changing gate
  # order so converted checkpoints remain numerically equivalent.
  for gate_name, gate_value in zip(('hi', 'hf', 'hg', 'ho'), _split_lstm_gates(hidden_kernel)):
    _set_path(params, f'{core_base}/{gate_name}/kernel', gate_value)
  for gate_name, gate_value in zip(('ii', 'if_', 'ig', 'io'), _split_lstm_gates(input_kernel)):
    _set_path(params, f'{core_base}/{gate_name}/kernel', gate_value)
  for gate_name, gate_value in zip(('hi', 'hf', 'hg', 'ho'), _split_lstm_gates(bias)):
    _set_path(params, f'{core_base}/{gate_name}/bias', gate_value)


def convert_state(source_state: dict) -> dict:
  config = _jax_config_from_tf_config(source_state['config'])
  policy = _policy_from_config(config)
  params = jax_utils.get_module_state(policy)
  reader = LeafReader(source_state['state']['policy'])

  _convert_autoregressive_head(params, reader)
  _convert_network(params, reader, config['network'])
  _skip_legacy_policy_value_readout(reader, config['network'])
  reader.finish()

  # TODO: Convert optimizer/value-function state.
  return {
      'state': {'policy': params},
      'config': config,
      'name_map': copy.deepcopy(source_state.get('name_map', {})),
      'step': int(source_state.get('step', source_state['state'].get('step', 0))),
  }


def convert_file(source_path: str | Path, output_path: str | Path) -> None:
  source_path = Path(source_path)
  output_path = Path(output_path)
  with source_path.open('rb') as f:
    source_state = pickle.load(f)
  converted = convert_state(source_state)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('wb') as f:
    pickle.dump(converted, f)


def _frames(path: Path) -> data.Frames:
  game = data.read_table(str(path), compressed=True)
  state_action = data.StateAction(
      game, game.p0.controller, np.zeros(len(game.stage), np.int32))
  is_resetting = np.zeros(len(game.stage), np.bool_)
  is_resetting[0] = True
  return data.Frames(state_action, is_resetting, reward.compute_rewards(game))


def _tf_logits(source_path: Path, game_path: Path):
  from slippi_ai.tf import saving as tf_saving

  policy = tf_saving.load_policy_from_disk(str(source_path))
  frames = _frames(game_path)
  frames = frames._replace(state_action=policy.network.encode(frames.state_action))
  frames = utils.map_nt(lambda x: np.expand_dims(x, 1), frames)
  outputs = policy.unroll(frames, policy.initial_state(1))
  return utils.map_nt(
      lambda x: np.squeeze(x.numpy(), 1)[::_VERIFY_SUBSAMPLE],
      outputs.distances.logits)


def _jax_logits(output_path: Path, game_path: Path):
  policy = saving.load_policy_from_state(saving.load_state_from_disk(str(output_path)))
  frames = _frames(game_path)
  frames = frames._replace(state_action=policy.network.encode(frames.state_action))
  frames = utils.map_nt(lambda x: jnp.expand_dims(jnp.asarray(x), 1), frames)
  outputs = policy.unroll(frames, policy.initial_state(1))
  logits = utils.map_nt(
      lambda x: np.squeeze(np.asarray(x), 1)[::_VERIFY_SUBSAMPLE],
      outputs.distances.logits)
  return policy.controller_head.controller_embedding, logits


def _sum_leaves(xs):
  total = None
  for x in jax.tree.leaves(xs):
    total = x if total is None else total + x
  return total


def _verify_game(source_path: Path, output_path: Path, game_path: Path) -> dict[str, float]:
  tf_logits = _tf_logits(source_path, game_path)
  embedding, jax_logits = _jax_logits(output_path, game_path)
  kl_tf_jax = _sum_leaves(embedding.map(
      lambda e, p, q: e.kl_divergence(jnp.asarray(p), jnp.asarray(q)),
      tf_logits, jax_logits))
  kl_jax_tf = _sum_leaves(embedding.map(
      lambda e, p, q: e.kl_divergence(jnp.asarray(p), jnp.asarray(q)),
      jax_logits, tf_logits))
  max_abs = max(
      float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
      for a, b in zip(tree.flatten(tf_logits), tree.flatten(jax_logits)))
  return {
      'mean_kl_tf_to_jax': float(jnp.mean(kl_tf_jax)),
      'max_kl_tf_to_jax': float(jnp.max(kl_tf_jax)),
      'mean_kl_jax_to_tf': float(jnp.mean(kl_jax_tf)),
      'max_abs_logit_diff': max_abs,
  }


def verify_file(source_path: str | Path, output_path: str | Path, data_dir: str | Path) -> None:
  source_path = Path(source_path)
  output_path = Path(output_path)
  game_paths = sorted(path for path in Path(data_dir).rglob('*') if path.is_file())
  game_paths = game_paths[:_VERIFY_MAX_GAMES]
  if not game_paths:
    raise ValueError(f'no verification games found under {data_dir}')

  stats = [_verify_game(source_path, output_path, path) for path in game_paths]
  print(f'verify_data_dir: {data_dir}')
  print(f'verify_games: {len(game_paths)}')
  for key in stats[0]:
    values = [item[key] for item in stats]
    print(f'{key}: mean={np.mean(values):.6g} max={np.max(values):.6g}')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('source', help='legacy TensorFlow pickle checkpoint')
  parser.add_argument('output', help='output JAX pickle checkpoint')
  parser.add_argument(
      '--verify-data-dir',
      default=str(paths.TOY_DATA_DIR),
      help='run post-conversion TF/JAX parity on parsed game tables (defaults to toy data)')
  args = parser.parse_args()

  convert_file(args.source, args.output)
  verify_file(args.source, args.output, args.verify_data_dir)


if __name__ == '__main__':
  main()
