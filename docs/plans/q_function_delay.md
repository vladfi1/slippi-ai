# Plan: delay support for the JAX Q-function code

Status: all three phases done (2026-09-04, 2026-09-06, 2026-09-07).

Phase 1 landed `data.delayed_frames`, used it from `Policy.imitation_loss` and
`q/q_fn_learner.py`, added the divisibility check in `q/train_q_fn.py`, and
added `tests/delayed_frames_test.py`. Smoke tests `q/tests/train_q_fn.py` and
`jax/tests/train_policy_test.py` pass at delay 0 and 3.

Phase 2 swapped the stubs in `q/q_policy_learner.py` and
`q/compare_q_functions.py` for the helper and added `override_delay` to
`q/train_q_policy.py` and `q/compare_q_functions.py`. The comparator was also
ported to the epinet Q-function API (it still called the removed
`loss_and_action_state`) and now scores both q-functions with the same
epistemic indices so that self-comparison is exact. Verified by training a
delay-3 toy Q-function with `q/tests/train_q_fn.py --config.delay=3` and
feeding its checkpoint to `q/tests/train_q_policy.py` and
`q/tests/compare_q_functions.py` with `--config.override_delay=3`; both also
reject a delay-0 Q-function at delay 3, and the Q-policy run restores from its
own checkpoint.

Phase 3 gave `FrameSkipConverter` a `skip_delay` overlap buffer, switched
`q/train_q_rl.py` to that converter (it previously had its own copy of the
frame-skip conversion, which did not zero rewards across game boundaries),
applied `data.delayed_frames` once in `q/rl_learner.py` and carried the sample
policy's hidden state in the learner. `tests/frame_skip_converter_test.py`
checks that consecutive delayed windows are contiguous and that actions,
rewards and actor logits line up with `data.delayed_frames`. Verified with
`q/tests/train_q_rl.py` at delay 0 and, with a delay-3 toy Q-function
checkpoint and `--config.override_delay=3`, at delay 3.

## Semantics

The imitation policy defines the convention in `slippi_ai/jax/policies.py`
(`Policy.imitation_loss`). With delay `D` frames and frame skip `FS`, the
skip-delay is `Ds = D // FS` (delay must be divisible by frame skip). From a
chunk of `U + Ds + 1` frame-skip steps it keeps

- states `[0, U]`
- actions `[Ds, U + Ds]`
- rewards `[Ds, U + Ds - 1]`

The Q-function adopts exactly this alignment. It then models the value of
state `t` given the actions already committed up to `t + D`, and scores the
candidate action for `t + D + 1` on rewards from `r(t + D)` onward.

This is the right target for the argmax step: the Q-policy picks its next
action from the information it has at time `t`, so the Q-function must be
conditioned on that same information set rather than on the future state where
the action lands. Dropping rewards between `t` and `t + D` is harmless for
ranking since none of them depend on the candidate action.

Existing stubs (`_get_delayed_frames` in `q/q_fn_learner.py`,
`q/q_policy_learner.py`, `nash/q_fn_learner.py`, `nash/nash_policy_learner.py`)
already do this slicing behind `assert delay == 0`. Two problems beyond the
assert:

- They slice by the raw frame delay, not `delay // frame_skip`.
- The slicing is copy-pasted. Pull it into one helper
  (`data.delayed_frames`) shared by the imitation policy and all learners.

## Phase 1: offline Q-function

Files: `slippi_ai/jax/q/q_fn_learner.py`, `slippi_ai/jax/q/train_q_fn.py`.

- Replace the stub with the shared helper, store the skip-delay, validate
  divisibility.
- `q_function.py` needs no change: its loss takes `T + 1` states and `T`
  rewards, and the delayed slice preserves that. The overlap passed to the data
  source (`extra_frames = delay + frame_skip`) is already correct.
- Keep the permissive behavior where an explicit `delay` may differ from the
  compatible policy's delay (useful for tests with the delay-0 toy checkpoint).
- Tests: existing smoke test with `--config.delay=3` (toy policy has
  frame skip 3, delay 0). Add a numpy unit test of the helper checking reward
  index `t` maps to original index `t + Ds`.

## Phase 2: offline Q-policy

Files: `slippi_ai/jax/q/q_policy_learner.py`, `slippi_ai/jax/q/train_q_policy.py`,
`slippi_ai/jax/q/compare_q_functions.py`.

- Same swap; remove the asserts. Sample and Q policies already unroll on the
  delayed frames via the same path as imitation, so their outputs at index `t`
  and the Q-function's action state at index `t` both refer to the action at
  `t + D + 1`.
- Testing needs a delayed imitation checkpoint. Delay does not change the
  architecture, so add an `override_delay` option to the Q-policy trainer
  (mirroring `train_q_rl.py`) and chain a delay-3 Q-function checkpoint into a
  delay-3 Q-policy run.
- Inference is free: the delayed agent wrapper in `eval_lib.py` already queues
  actions for any policy with delay.

## Phase 3: online Q RL

Files: `slippi_ai/jax/q/rl_learner.py`, `slippi_ai/jax/q/train_q_rl.py`, and
`FrameSkipConverter` in `slippi_ai/jax/rl/learner.py`.

Rollouts are contiguous windows with no overlap, so applying the delayed slice
directly would shorten each rollout by `Ds` steps and leave carried hidden
states `Ds` steps behind the next rollout. Use an overlap buffer rather than
the `delayed_actions` route sketched in `rl/learner.py:get_delayed_frames`:

- Have the frame-skip converter retain the trailing `Ds` frame-skip steps of
  the previous trajectory (sampled outputs, reset flags, rewards) and prepend
  them to the next one, like it already does for frame skip's trailing actions.
- Apply the delayed slice once to the whole frame-skip trajectory. Every
  downstream unroll (including actor-KL logits) then works unchanged and the
  Q-function loses no training steps.
- Carry the sample policy's hidden state in the learner like the other modules
  instead of reading it from the trajectory.
- Drop the `delayed_actions` asserts; the queued actions are exactly the ones
  the next rollout contains.
- Mid-rollout resets contaminate the `Ds` steps before each game boundary with
  next-game rewards. A validity mask can be a follow-up.
- The first rollout needs padding for the buffer; burn-in absorbs the one
  garbage window.
- Test with the RL smoke test plus `override_delay`; requires a delay-3 toy
  Q-function checkpoint.

## Follow-ups and validation

- The nash learners have identical stubs and the same frame-skip bug; port them
  once the helper exists.
- Validation: train a Q-function at netplay-typical delay on real data; expect
  value loss to rise vs delay 0 while `uev_delta` stays positive. Then train a
  Q-policy at that delay and evaluate against its teacher at the same delay.
