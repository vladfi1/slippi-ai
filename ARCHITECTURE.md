# Project Structure

`slippi-ai` trains Super Smash Bros. Melee agents from Slippi replays. The core
workflow is **imitation learning** from human replays, with **reinforcement
learning** (self-play and Q-learning variants) layered on top. The active stack
is JAX + Flax (`flax.nnx`); a legacy TensorFlow stack is kept for reference.

This document is a map for navigating the codebase, not an exhaustive reference.
Paths are relative to the repo root.

## Top-level layout

| Path | Purpose |
|------|---------|
| `slippi_ai/` | Main Python package (models, data pipeline, training, eval). |
| `slippi_ai/jax/` | JAX/Flax implementation — the primary, actively-developed stack. |
| `slippi_ai/tf/` | Legacy TensorFlow implementation. |
| `slippi_db/` | Replay parsing/preprocessing into the training dataset. |
| `scripts/`, `run_scripts/` | One-off and launch scripts. |
| `tests/` | Top-level tests (e.g. `tests/data_test.py`). |
| `skypilot/`, `docker/` | Cloud launch + container configs. |
| `notebooks/` | Scratch analysis notebooks (not part of the package). |

## Core package (`slippi_ai/`, framework-agnostic)

These modules define the data and domain types shared by both the JAX and TF
stacks:

- `types.py` — central type system: the `Game` / `StateAction` / `Frames`
  NamedTuples and helpers (`array_from_nt`, `game_array_to_nt`).
- `data.py` — data pipeline: `ReplayInfo`, `DataSource`, `Batch`,
  `BatchWithMeta`, `DatasetConfig`/`DataConfig`, and `build_sources`.
  **`Batch.to_frames(frame_skip)` is the frame-skip entry point** — it subsamples
  states, sums rewards over each frame-skip window, and turns the per-frame
  controllers into a `list[Action]` of length `frame_skip`.
- `utils.py` — nested-tuple utilities (`map_nt`, `batch_nest_nt`, `cached_*`).
- `nametags.py` — player-name normalization and name→code maps.
- `observations.py`, `reward.py`, `controller_lib.py` — observation config,
  reward shaping, controller helpers.
- `embed.py` (in `jax/`) / per-stack embeds — controller/state embeddings.
- `dolphin.py`, `envs.py`, `eval_lib.py`, `evaluators.py`, `agents.py` —
  live evaluation against the Dolphin emulator.
- `saving.py`, `paths.py`, `flag_utils.py` — checkpoint IO, canonical paths
  (including test checkpoints), and absl/fancyflags helpers.

## JAX stack (`slippi_ai/jax/`)

Shared building blocks:

- `embed.py` — embeddings. `get_state_action_embedding(..., frame_skip)` wraps
  the action in a `ListEmbedding(embed_action, frame_skip)`, which is why a
  frame-skipped action is a `list[Action]`.
- `networks.py` — recurrent/transformer cores. `build_embed_network(...,
  frame_skip, ...)` builds the state/action encoder; `construct_network` builds a
  bare RNN. Networks expose `unroll` / `scan` / `step` / `step_with_reset`.
- `controller_heads.py` — autoregressive controller heads. With frame skip,
  `sample` / `distance` / `distance_outputs` take and return **per-frame-skip
  lists** (one entry per action in the window).
- `policies.py` — `Policy` (network + controller head) with `delay` and
  `frame_skip`; `unroll_with_outputs` produces per-step outputs + imitation loss.
- `jax_utils.py` — sharding (`data_parallel_train`, `shard_map_loss_fn`, `PS`,
  `DATA_AXIS`), `MLP`, `lax_map`, dtype helpers.
- `rl_lib.py` — returns/discounts: `discount_from_halflife(halflife, frame_skip)`
  and `generalized_returns_with_resetting(..., lambda_)`.

Imitation learning (the base case):

- `train_policy.py` + `policy_learner.py` — policy imitation training. The
  `TrainManager.fetch_batch` calls `batch.to_frames(frame_skip)` then
  `network.encode`; this is the canonical single-player frame-skip data flow.
- `train_lib.py` — shared training harness + combined policy/value training.
- `value_function.py` + `vf_learner.py`, `train_vf.py` — single-player value
  function (good reference for frame-skip `unroll` over list-actions).

RL / Q-learning subpackages:

- `jax/q/` — **single-player Q-learning** (this is the non-nash variant).
  - `q_function.py` — `QFunction` + `QFunctionConfig`. A `core_net` produces a
    value per step; a separate `action_net` (initialized from the core output via
    `action_init`) is unrolled over the `frame_skip` actions to produce Q-values.
  - `q_fn_learner.py` / `train_q_fn.py` — train the Q-function alone.
    `compatible_policy` syncs embed/observation/`frame_skip` from a policy so the
    Q-function's action representation matches.
  - `q_policy_learner.py` / `train_q_policy.py` — distill a policy toward the
    argmax of the Q-function over sampled actions.
- `jax/nash/` — **two-player Nash Q-learning** (the up-to-date reference the `q/`
  code mirrors). `q_function.py` (two-player, with `to_merged_outputs`),
  `nash.py`/`optimization.py` (the equilibrium solvers), `nash_policy_learner.py`,
  `rl_learner.py`, and `train_*` scripts. `slippi_ai/nash/data.py` provides the
  `TwoPlayerDataSource`.
- `jax/rl/` — self-play RL (`run_lib.py`, `train_two.py`).
- Each subpackage has `tests/` (smoke tests on toy data) and `scripts/` (launch
  configs); `experiments/` holds run outputs.

## TF stack (`slippi_ai/tf/`)

A parallel, older implementation (`networks.py`, `policies.py`, `q_function.py`,
`learner.py`, `train_lib.py`, …). Not the focus of current development.

## Data preprocessing (`slippi_db/`)

Parses `.slp` replays (via libmelee / peppi) into the zlib-compressed parquet
dataset consumed by `slippi_ai/data.py`. See `run_preprocessing.py` and the
`parse_*.py` modules.

## Frame skip in one paragraph

With `frame_skip = k`, the agent acts every `k` frames. `Batch.to_frames(k)`
collapses each window of `k` raw frames into one step whose `state` is the last
frame, whose `reward` is the window sum, and whose `action` is the `list` of the
`k` controllers in the window. Discounts are adjusted via
`discount_from_halflife(halflife, frame_skip=k)`. Models embed the action list
with `ListEmbedding`, and Q-values for a window are computed by unrolling a small
action RNN over the `k` actions. Unroll lengths must be divisible by `k`, and
`config.data.random_offset = k` is set so windows are randomized across epochs.
