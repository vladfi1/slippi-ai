# Plan: delay support for nash q-policy training via action chains

Status: Phases 1-3 implemented (2026-09-15) and the restructuring
follow-ups done (2026-09-16); Phase 4 is proposed. Builds on
`docs/plans/q_function_delay.md`, which ported the delayed-frames alignment to
`nash/q_fn_learner.py`.

## The chain game

With frame delay `D` and frame skip `FS`, let `Ds = D // FS`. Under the
delayed alignment (`data.delayed_frames`) a network's input at index `t` is
`(s_t, a_{t+Ds})`, so at index `t` it knows states up to `t` and its own
actions up to `t + Ds`, and it chooses action `t + Ds + 1`.

A delayed player does not know the opponent's queued actions
`[t + 1, t + Ds]`. The one-step game over `a_{t+Ds+1}` scored by the delayed
q-function as trained today conditions on both players' real queues, so its
nash gives each player information it does not have. Instead, the game at
index `t` is over *chains*: with common knowledge `(s_{<=t}, a^1_{<=t},
a^2_{<=t})`, each player simultaneously picks `a_{t+1..t+Ds+1}`. With
`num_samples = N`, each player samples `N` chains from the sample policy
(plus the chain actually taken), the `(N + 1) x (N + 1)` chain-pair payoff
matrix is solved for a nash mixed strategy, and the nash policy regresses to
it, exactly as the current learner does for single actions.

This is a commitment approximation: in reality `a_{t+1..t+Ds}` were chosen
earlier with less information and only `a_{t+Ds+1}` is chosen at `t`. The
chain game lets both players re-plan the window at `t` but hides each other's
plan, which is the right information structure for the last action. With
`Ds = 0` everything reduces to the current code, which is the regression
test.

## Scoring, sampling and training by re-running the networks

All three networks keep the delayed alignment and their current
architectures. A chain `a'_{t+1..t+Ds+1}` is scored or sampled by restarting
a network from its hidden state at index `t - Ds` and re-running it over the
real states `s_{t-Ds+1..t}` paired with the sampled actions instead of the
real ones (input at index `j` is `(s_j, a'_{j+Ds})`), honouring the real
`is_resetting` flags. Every network's `scan` already returns the per-index
hidden states this needs.

- **Q-function.** Per player and sample: re-run the core net for `Ds` steps
  from `h_{t-Ds}` with the sampled prefix, take `action_init(core_out_t)`, and
  unroll the action net over `a'_{t+Ds+1}` as today. The two players' features
  are merged pairwise through the heads afterwards, so the cost is
  `N * Ds` core steps per player and index, not `N^2`
  (`multi_index_q_values_from_action_state` already has this structure for the
  action net). The result is `Q(s_{<=t}, a_{<=t}; chain^1, chain^2)`; nothing
  about the real queues survives.
- **Sample policy.** From `h_{t-Ds}` the output at index `t - Ds` predicts
  `a_{t+1}`: sample `a'_{t+1}` (prev `a_t`), step with `(s_{t-Ds+1},
  a'_{t+1})`, sample `a'_{t+2}`, ..., step with `(s_t, a'_{t+Ds})`, sample
  `a'_{t+Ds+1}`. That is `Ds` network steps and `Ds + 1` head samples per
  chain, and an exact sample of the delayed imitation policy's distribution
  over its next `Ds + 1` actions given the real states. For the opponent this
  is precisely the model of the unknown queue we want.
- **Nash policy.** The log-probability of a chain is the sum of the one-step
  log-probs along the same re-run, so the network sees the sampled prefix at
  every step. Cross-entropy to the nash mixture over chains as before. At
  inference the policy acts one step at a time with its real queue, which is
  the induced chain distribution it was trained on, so no agent or
  `eval_lib` changes are needed.

## Phase 1: q-function with prefix rewards (done)

Files: `slippi_ai/data.py`, `slippi_ai/jax/nash/q_fn_learner.py`,
`slippi_ai/jax/nash/train_q_fn.py`, `tests/delayed_frames_test.py`.

The delayed q-function's target starts at reward `t + Ds`. For a one-step
game the dropped rewards `[t, t + Ds - 1]` are a constant across the payoff
matrix, but across chains they vary (a prefix that walks into a hit loses that
damage reward), so a chain scorer must keep them. `data.delayed_frames` gets a
`keep_prefix_rewards` option that pairs index `t` with reward `t` (rewards
`[0, U - 1]`) instead of `t + Ds`; the states, actions and reward count are
unchanged, so `QFunction` needs no change. The nash learner always keeps
the prefix rewards (the chain game is its only delayed consumer, and at delay
0 the two conventions coincide); the non-nash `q/` learners keep the default.

The q-target at index `t` is then `r_t + g V_{t+1}` (at `gae_lambda = 0`),
and because `V_{t+1}` conditions on actions up to `t + Ds + 1` it already
sees the whole chain: no longer bootstrap horizon is needed. The bootstrap at
index `U` is the value given actions up to `U + Ds`, regressing to the return
from `U`, which is consistent. A side benefit is that the reset mask
`is_resetting[1:]` now lines up exactly with the rewards, as at delay 0.

Verified with `nash/tests/train_q_fn.py --config.delay=3` and the unit
test. Expect a somewhat higher value loss than without prefix rewards on
real data, since the prefix rewards depend on the not-yet-observed states;
`uev_delta` should stay positive.

## Phase 2: chain machinery in the nash policy learner (done)

Files: `slippi_ai/jax/nash/nash_policy_learner.py`,
`slippi_ai/jax/nash/q_function.py`, `slippi_ai/jax/nash/utils.py`,
`slippi_ai/jax/policies.py`.

- Learner: swap the `_get_delayed_frames` stub for the shared helper with
  `keep_prefix_rewards=True`, store `skip_delay`, check divisibility.
- Per-index hidden states: the sample policy, q-function core net and nash
  policy unrolls use `scan` instead of `unroll` so that `h_{t-Ds}` is
  available for every `t`. For the q-function, `loss_and_action_state` also
  returns the per-index core states (alongside `action_init_state`).
- Chunk boundary: `h_{t-Ds}` for `t < Ds` lies in the previous chunk, so a
  chunk's first `Ds` indices cannot host a game. The trainer therefore
  overlaps consecutive chunks by `Ds` extra steps (`extra_frames = frame_skip
  + 2 * delay`): after the delayed slice a chunk has `T = U + Ds` steps, the
  networks unroll over all of them, the game and every loss cover the `U`
  valid indices `[Ds, T)`, and the learner carries the hidden state after
  index `U - 1`, where the next chunk starts. Nothing is lost and no index is
  counted twice; the cost is `Ds / U` extra network steps per chunk.
- Re-run helper (`nash_utils.rerun_from`): given per-index states, a window
  of real states `s_{t-Ds+1..t}` and a chain prefix, run the network for `Ds`
  steps and return the outputs at indices `t - Ds .. t`. The window inputs are
  sliding slices over the time axis (`[Ds, T', B, 2]`), which for `Ds = 2`
  is a small memory multiple of the state embeddings. Batched over
  `T' x B x 2` as the batch and vmapped (or `lax_map`ped with
  `sample_batch_size`) over the `N + 1` samples.
- Chain sampler on top of the re-run helper for the sample policy
  (`_unroll_sample_policy`) and for the nash policy's own samples in the
  `compute_nash_policy_qs` diagnostics. The samples become a
  `(Ds + 1) * FS` action list, `[S, T', B, 2]` per element, which
  `compute_unique_fraction` and the existing sample-axis logic accept
  unchanged. The action-taken sample is the real chain (sliding windows over
  the chunk's actions; `data.action_windows` helper).
- Chain scoring in `_unroll_q_function`: per sample and player, re-run the
  core net over the prefix, then reuse `multi_index_q_values_from_action_state`
  with per-sample `action_init_state` (`[S, T', B, 2, H]` instead of
  `[T', B, 2, H]`; small generalisation of that function).
- `_compute_nash`: unchanged.
- Chain log-probs in `_unroll_nash_policy`: re-run the nash policy over the
  prefix per sample with gradients, sum the `Ds + 1` head distances, and feed
  the existing cross-entropy. Use `nnx.remat` and `sample_batch_size` as the
  distance function already does; memory is `N * (Ds + 1)` policy steps per
  index.
- Reset handling: a chain window that crosses `is_resetting` mixes games.
  Mask indices whose window `[t - Ds + 1, t + Ds + 1]` contains a reset out of
  the nash-policy loss (the q-function re-run still honours the flags so the
  values stay finite).

Implementation notes: `Policy.scan_with_outputs` and
`QFunction.scan_core` expose the per-index hidden states;
`nash_utils.ChainContext.rerun` is the one re-run loop, over which
`sample_chain`, `chain_log_prob` and `QFunction.chain_action_init_state`
are thin wrappers, and `multi_index_q_values_from_action_state(
per_sample_init=True)` scores the last actions from per-sample core states.
At `Ds = 0` the re-run is empty, so the same code path serves every delay.
`QFunction.scan_core` returns the per-step values and `ensemble_outputs`
turns a slice of them into the loss, so the learner computes the q-function
loss on the valid indices only and slices the imitation losses the same
way. `nash_utils.ChunkLayout` holds the chunk geometry (see the follow-ups).

Validation: `tests/nash_chain_test.py` checks the slicing alignment on
index-valued frames and that re-running the toy policy over the chain it
actually took reproduces the main unroll's per-step log-probs (skip-delay 2).
The `Ds = 0` toy run is not bit-for-bit comparable across runs because the
data source randomizes window offsets; its metrics are in the same range as
before the change.

## Phase 3: trainer plumbing (done)

Files: `slippi_ai/jax/nash/train_nash_policy.py`.

- Keep the `q_fn_config.delay == imitation_config.policy.delay` check.
  Nash q-function checkpoints from before 2026-09-15 with nonzero delay were
  trained on dropped prefix rewards and must not be used for chains.
- No `override_delay` (unlike `train_q_rl.py`): the policies are trained at
  the imitation checkpoint's delay, which must match the q-function's. The
  tests use `checkpoints/fs_demo` (delay = frame_skip = 3, made with
  `jax/scripts/create_policy_checkpoint.py --config.policy.delay=3`) and the
  toy q-functions trained against it.
- `extra_frames = frame_skip + 2 * delay` (`ChunkLayout.extra_frames`, which
  the trainer asks the learner for): the chunk has `U + 2 Ds + 1` steps, the
  delayed slice keeps states `[0, U + Ds]`, actions `[Ds, U + 2 Ds]` and
  rewards `[0, U + Ds - 1]`, and the game covers indices `[Ds, U + Ds)`.
- The q-function is passed to `run_nash_policy_qs` as its module rather
  than closed over by the nash-policy unroll (as it was before delay), so
  its state is traced and can be updated in place, which is what Phase 4
  needs to train it alongside the nash policy.

Validation: `nash/tests/train_q_fn.py` and `nash/tests/train_nash_policy.py`
(the `fs_demo` policy and its delay-3 toy nash q-function) train and evaluate
on CPU (skip-delay 1); `nash/scripts/eval_nash_q.py --toy_data` evaluates the
same pair and rejects a q-function whose delay differs from the imitation
policy's; `nash_cross_entropy` is about twice the
delay-0 value, as expected for two-action chains; the run restores from its
own checkpoint.

## Phase 4 (follow-up): online nash RL

`nash/rl_learner.py` and `train_nash_rl.py` assert `not
trajectory.delayed_actions`. With the actor's queued actions folded into the
trajectory as the PPO/Q-RL converters do, `get_delayed_frames` gives the same
delayed alignment and the Phase 2 machinery applies unchanged. With prefix
rewards every one of the `T` rewards pairs with a state, so unlike the Q-RL
case no transitions are lost; only the bootstrap needs the queued actions.

## Restructuring follow-ups (done 2026-09-16)

The chain machinery worked but spread the same ideas over several places.
In rough order of payoff:

1. **One re-run loop.** `nash_utils.sample_chain`, `nash_utils.chain_log_prob`
   and `QFunction.chain_action_init_state` were the same loop: start from the
   per-index states, step a network over the context's inputs with the
   chain's actions, and do something with each output. Done:
   `ChainContext.rerun(network, act)` steps the network over a chain chosen
   one action at a time by `act(k, outputs, prev_action)` and returns the
   `Ds + 1` per-index `(outputs, prev_action)` steps (`rerun_chain` for a
   known chain); the three are wrappers over it, and the q-function's
   version takes the `ChainContext` built from its `scan_core` outputs
   instead of its own `inputs` and `resets` arguments. A generator was
   considered but the sampler needs to feed each sampled action back, which
   a callback expresses more plainly than `send`.
2. **Drop the `Ds = 0` special case in the q-function unroll.** Done: with
   an empty prefix `chain_action_init_state` returns the action-init state
   of the context's outputs, the unroll always uses `per_sample_init=True`,
   and the matching branch in the nash-policy diagnostics is gone. The
   q-function unroll passes its per-step core outputs and states through
   (`QFunctionOutputs.core_outputs`, `core_states`) so the diagnostics can
   rebuild the context. `tests/nash_chain_test.py` checks that the toy
   q-function's re-run over the chain actually taken reproduces its scan's
   action-init states at `Ds = 0` and `2`.
3. **A `ChunkLayout` object.** Done: `nash_utils.ChunkLayout(delay,
   frame_skip)` offers `extra_frames`, `delayed_frames`, `num_valid`,
   `game_slice`, `carried_state`, `context` and `taken_chain`; the learner
   holds one and the trainer asks it for the chunk overlap. The RL learner
   can reuse it (Phase 4).
4. **Name the q-function unroll's outputs.** Done: `QFunctionOutputs` with
   a parallel `QFunctionOutputSpecs`. Note jax only matches a pytree prefix
   of the same named-tuple type, so the specs are converted with
   `as_prefix()` where they serve as an input spec.
5. **Split `_unroll_nash_policy`.** Done: `_nash_targets` builds the
   regression target (nash mixture, subsampling, diagnostics of the nash
   solution) into a `NashTargets` tuple and `_chain_cross_entropy` scores
   the chains against it; the nash-policy-versus-nash diagnostic was already
   its own method (item 6). `nash_utils.nash_solution_probs` and
   `nash_utils.merge_players` hold the shared pieces.
6. **Score the nash policy's chain in a q-function step.** Done
   (2026-09-16): the nash-policy loss returns its sampled chain and
   `_nash_policy_qs`, a separate sharded function on the q-function, scores
   it. This was forced rather than optional: the q-function's `ControllerRNN`
   embedding unrolls with `nnx.scan`, which cannot run on a closed-over module
   inside another module's loss (or under a raw `jax.lax.map`), so the
   per-sample core re-run in the q-function unroll also had to become an
   nnx-aware map with the q-function as an argument. The diagnostic now uses
   all S samples regardless of `subsample`.
7. **Measure `scan` versus `unroll`.** Done: both lower to the same
   per-step `nnx.scan` (`jax_utils.dynamic_rnn` and `scan_rnn`); `scan`
   only additionally stacks the per-step states. Measured on an RTX PRO
   5000 (`tx_like`, LSTM, 84 steps, batch 256 or 512 x 2 players, hidden
   1024 x 1 layer or 512 x 2 layers): forward and gradient times agree
   within noise (for example 43 vs 42 ms forward and 121 vs 121 ms with
   gradients at hidden 1024, batch 512). Nothing to recover here.

## Issues and risks

1. **Commitment approximation.** The nash re-plans `a_{t+1..t+Ds}` at `t`
   although they were committed earlier. The nash policy's one-step
   conditional given its real queue is the marginal of its induced chain
   distribution, so the executed behaviour is consistent with training, but
   the equilibrium is of the re-planning game, not of the true
   imperfect-information game.
2. **Cost.** Per index and player: `N * Ds` core-net steps for the
   q-function, `N * (Ds + 1)` sample-policy steps (sequential within a chain),
   and `N * (Ds + 1)` nash-policy steps with gradients. The core nets are the
   large networks, so at `Ds = 2` and `N = 8` the learner does roughly 16x
   the network work of the delay-0 learner outside the solver; the
   `(N + 1)^2` nash solve is unchanged. `sample_batch_size` and remat bound
   memory; the per-index hidden states add `T * B * 2 * H` per network.
3. **Chunk boundary.** Handled by the extra overlap; the alternative of
   carrying the last `Ds` per-index states and inputs of every network across
   chunks would save the `Ds / U` recomputation at the cost of a much larger
   carried state.
4. **Reset contamination.** Windows that straddle a game boundary are masked
   from the nash-policy loss; about `2 * Ds / U` of indices near boundaries.
5. **Off-distribution scoring.** The q-function is evaluated on sampled
   prefixes, not just sampled last actions. The sample policy is an imitation
   policy so the chains are realistic, but the epinet's epistemic spread
   should be watched as chain length grows.
6. **Checkpoint compatibility.** Keeping the prefix rewards changes the
   value semantics (return from `t` rather than `t + Ds`) for every delayed
   nash q-function trained from 2026-09-15 on; earlier delayed nash
   checkpoints are not usable for chains. The non-nash `q/` learners keep
   the dropped-rewards convention, where it is harmless.

## Alternatives considered

- **Chain q-function** (action net unrolled over the whole `(Ds + 1) * FS`
  chain from a delay-0 core state, with delay-0 policies whose heads are
  iterated on a fixed output). Cheaper per sample but needs `Ds + 1`-step
  bootstrap targets (the tail of the chain is invisible to a one-step
  target), produces less realistic sample chains, limits the nash policy to
  a Markov-in-previous-action chain distribution, and needs a new agent mode
  at inference. Superseded by the re-run design.
- **Delayed policies trained on the chain nash's last-action marginal**
  without re-running. Cheapest to plumb, but the target ignores the policy's
  real prefix while the network conditions on it. Rejected.
