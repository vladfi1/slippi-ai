"""Compare fp32 vs bf16 JAX agent logits on toy data via KL divergence."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tree
import tqdm

from slippi_ai import data, paths, reward, utils
from slippi_ai.jax import jax_utils, saving, policies

_UNROLL_LENGTH = 80


def _frames(path: Path) -> data.Frames:
  game = data.read_table(str(path), compressed=True)
  state_action = data.StateAction(
      game, game.p0.controller, np.zeros(len(game.stage), np.int32))
  is_resetting = np.zeros(len(game.stage), np.bool_)
  is_resetting[0] = True
  return data.Frames(state_action, is_resetting, reward.compute_rewards(game))


def _make_unroll_fn(policy: policies.Policy):
  @jax_utils.nnx_jit
  def run(policy: policies.Policy, frames: data.Frames, initial_state):
    frames = utils.map_nt(lambda x: jnp.expand_dims(jnp.asarray(x), 1), frames)
    outputs = policy.unroll(frames, initial_state)
    return (
        utils.map_nt(lambda x: jnp.squeeze(x, 1), outputs.distances.logits),
        outputs.final_state,
    )

  cached_run = jax_utils.cached_functional_jit(run, policy)

  def run_with_encode(frames: data.Frames, initial_state):
    frames = frames._replace(state_action=policy.network.encode(frames.state_action))
    return cached_run(frames, initial_state)

  return run_with_encode


def _sum_leaves(xs):
  total = None
  for x in jax.tree.leaves(xs):
    total = x if total is None else total + x
  return total


def _compare_game(
    fp32_unroll,
    bf16_unroll,
    fp32_policy: policies.Policy,
    bf16_policy: policies.Policy,
    game_path: Path,
    unroll_length: int,
) -> dict[str, np.ndarray]:
  frames = _frames(game_path)
  num_frames = len(frames.is_resetting)
  embedding = fp32_policy.controller_head.controller_embedding

  bf16_state = bf16_policy.initial_state(1)

  all_fp32_logits = []
  all_bf16_logits = []

  chunk_starts = range(0, num_frames - 1, unroll_length)
  for start in tqdm.tqdm(chunk_starts, desc=game_path.name, leave=False):
    end = min(start + unroll_length + 1, num_frames)
    if end - start < 2:
      break
    chunk = jax.tree.map(lambda x: x[start:end], frames)

    fp32_state = jax_utils.cast_floats_to_dtype(bf16_state, jnp.float32)
    fp32_logits, _ = fp32_unroll(chunk, fp32_state)
    bf16_logits, bf16_state = bf16_unroll(chunk, bf16_state)

    all_fp32_logits.append(utils.map_nt(np.asarray, fp32_logits))
    all_bf16_logits.append(utils.map_nt(lambda x: np.asarray(x, dtype=np.float32), bf16_logits))

  fp32_logits = jax.tree.map(lambda *xs: np.concatenate(xs), *all_fp32_logits)
  bf16_logits = jax.tree.map(lambda *xs: np.concatenate(xs), *all_bf16_logits)

  kl_fp32_to_bf16 = np.asarray(_sum_leaves(embedding.map(
      lambda e, p, q: e.kl_divergence(jnp.asarray(p), jnp.asarray(q)),
      fp32_logits, bf16_logits)))
  kl_bf16_to_fp32 = np.asarray(_sum_leaves(embedding.map(
      lambda e, p, q: e.kl_divergence(jnp.asarray(p), jnp.asarray(q)),
      bf16_logits, fp32_logits)))

  per_leaf = []
  for a, b in zip(tree.flatten(fp32_logits), tree.flatten(bf16_logits)):
    diff = np.abs(np.asarray(a) - np.asarray(b))
    if diff.ndim > 1:
      diff = diff.mean(axis=tuple(range(1, diff.ndim)))
    per_leaf.append(diff)
  abs_logit_diff = np.mean(np.stack(per_leaf), axis=0)

  return {
      'kl_fp32_to_bf16': kl_fp32_to_bf16,
      'kl_bf16_to_fp32': kl_bf16_to_fp32,
      'abs_logit_diff': abs_logit_diff,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint', help='JAX checkpoint path')
  parser.add_argument(
      '--data-dir',
      default=str(paths.TOY_DATA_DIR),
      help='directory of game tables (defaults to toy data)')
  parser.add_argument(
      '--max-games', type=int, default=5,
      help='maximum number of games to evaluate')
  parser.add_argument(
      '--unroll-length', type=int, default=_UNROLL_LENGTH,
      help='frames per unroll; fp32 hidden state is reset from bf16 after each')
  args = parser.parse_args()

  state = saving.load_state_from_disk(args.checkpoint)
  fp32_policy = saving.load_policy_from_state(state)
  bf16_policy = jax_utils.cast_params_to_dtype(fp32_policy, jnp.bfloat16)

  print(f'fp32 dtype: {jax_utils.module_dtype(fp32_policy)}')
  print(f'bf16 dtype: {jax_utils.module_dtype(bf16_policy)}')

  fp32_unroll = _make_unroll_fn(fp32_policy)
  bf16_unroll = _make_unroll_fn(bf16_policy)

  game_paths = sorted(p for p in Path(args.data_dir).rglob('*') if p.is_file())
  game_paths = game_paths[:args.max_games]
  if not game_paths:
    raise ValueError(f'no games found under {args.data_dir}')

  print(f'evaluating {len(game_paths)} game(s), unroll_length={args.unroll_length}')
  stats = [
      _compare_game(fp32_unroll, bf16_unroll, fp32_policy, bf16_policy, p, args.unroll_length)
      for p in tqdm.tqdm(game_paths, desc='games')
  ]

  for key in stats[0]:
    values = np.concatenate([item[key] for item in stats])
    print(f'{key}: mean={np.mean(values):.6g} max={np.max(values):.6g}')


if __name__ == '__main__':
  main()
