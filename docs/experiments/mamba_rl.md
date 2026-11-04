# Mamba Samus self-play RL (60-second unrolls)

Status (2026-11-04): paused. RL diverges within ~50 steps; the leading
hypothesis is biased advantages from the value function. See "Next steps".

## Goal

Self-play RL for the Mamba2 Samus imitation model with infinite time,
60-second unrolls, 256 envs, 10000 steps, KL weights 1e-2, name Morsecode,
rating 3000.

* Teacher: `experiments/jax/mamba_unroll/samus_u3600/samus_d21_mamba3x1024s128_rating/latest.pkl`
  (Mamba2 3x1024, state 128, best imitation eval loss 0.6078).

## Value functions

Trained with `vf_tx_like.py` against the teacher (`--config.compatible_policy`),
Mamba2 core, `ffw_multiplier=2`, batch 24 x 3600 frames, 4 epochs
(`runtime.max_epochs`, added for this). Eval uses lambda=1 and train uses
one-step TD targets, so only eval losses are comparable.

| VF | Best eval loss | At epoch | Speed |
|---|---|---|---|
| Mamba2 1x768 | **0.1339** | 2.8 | ~2.9 epochs/h |
| Mamba2 2x768 | 0.1352 | 1.9 | ~1.9 epochs/h |

The 1x768 is used for RL:
`experiments/jax/vf_mamba/vf_samus_mamba1x768_u3600/vf_samus_tx1x768_enhanced-192_um1/latest.pkl`
(the tag in the path says `tx` but the network is Mamba2).

## RL command

```
python slippi_ai/jax/rl/scripts/launch_rl.py \
  --config.teacher <teacher above> \
  --config.value_function <1x768 VF above> \
  --config.actor.rollout_length 3600 --config.actor.num_envs 256 \
  --kl_weight 1e-2 \
  --config.agent.name Morsecode --config.agent.rating 3000 \
  --config.learner.microbatch_size 8 --config.learner.teacher_mbs 32 \
  --config.learner.value_mbs 16 \
  --config.learner.learning_rate 1e-5 \
  --config.learner.ppo.max_mean_actor_kl 3e-3 \
  --config.runtime.max_step 10000 \
  --config.runtime.expt_root experiments/jax/rl_mamba/<id> --wandb.name <id>
```

* Memory: these microbatch sizes fit with the fp16 policy. An fp32 policy
  OOMs in `ppo_epoch` at `microbatch_size=8`.
* Speed: ~20 s rollout (~70k env fps) + learner; ~26 s/step during value
  burnin, ~32 s with 1 PPO epoch, ~38 s with 2. 10000 steps ~ 4.4 days.

## Finding 1: mid-unroll resets (fixed)

Infinite time does not prevent resets: sim-env games still end on stocks,
about 45 resets per 3600 x 256 rollout (`actor.unexpected_reset` ~4.9e-5).
The Mamba2 unroll used to ignore resets after the first frame while the
actor's step honored them, so the learner's state was wrong for the rest of
the rollout after every reset. Symptom: actor KL ~5-7e-4 at step 0 (no
update yet) with per-frame maxima of 9-16 nats, starting at random points in
the rollout; fp32 halved it but did not fix it.

Fix: `Mamba2Layer.unroll` now honors all resets by masking terms that cross
a reset in the SSD (`_ssd_chunk`) and the short conv (`_mix`). After the fix
the per-frame max actor KL dropped to ~0.04.

## Finding 2: fp16 step-vs-unroll drift

With the reset fix, the pre-update actor KL is still ~1.3e-4 (fp16 actor
step vs fp16 learner unroll), just over the default
`ppo.max_mean_actor_kl=1e-4`. It grows over the rollout (~3e-6 in the first
frames). tx_like runs, whose step and unroll are the same computation, sit
around 1e-5 post-update. Hence `max_mean_actor_kl=3e-3` above.

## Finding 3: the policy diverges (open)

| Run | LR | Crashed at | Final actor KL |
|---|---|---|---|
| i9c6ueix | 3e-5 (launch default) | ~step 30 | 3.2e-3 |
| kmpc8y50 | 1e-5 | ~step 52 | 3.2e-3 |

kmpc8y50, post-update metrics:

| step | teacher KL | entropy | ppo_objective | max grad norm |
|---|---|---|---|---|
| 0 | 0.0007 | 0.65 | -0.053 | - |
| 15 | 0.0007 | 0.65 | -0.036 | 0.22 |
| 27 | 0.034 | 0.85 | -0.014 | 0.23 |
| 43 | 0.22 | 0.79 | +0.008 | 0.83 |
| 51 | 0.32 | 0.90 | +0.013 | 1.97 |

Steps 0-7 are value-only and 8-15 use one PPO epoch, so the drift starts with
the 2-epoch updates at step 16. For comparison, tx_like RL runs at lr 1e-5
and KL weight 1e-2 (s3ut255n, 5th5l3mb) plateau at teacher KL ~0.1 after
50k+ steps. Lowering the LR only delayed the divergence.

Hypothesis: biased advantages. At step 0 there is no update, so the ratios
are 1 and `ppo_objective` is the mean advantage: -0.053. Self-play is
zero-sum, so it should be ~0. A negative mean advantage lowers the
probability of every sampled action, which raises entropy and moves the
policy away from the teacher, matching what we see. The value loss is small
(uev ~0.007), so the VF may fit with a constant offset rather than fail.
Candidate causes:

* Reward/discount mismatch between VF training (`reward_halflife`, damage
  terms in the VF checkpoint's config) and the RL learner config.
* The VF was trained on human data; sim-env self-play with infinite time and
  stocks may be out of distribution.

Ruled out: microbatched gradients are averaged (`take_mean`), not summed.

## Next steps

1. Diff the reward/discount config in the VF checkpoint against the RL
   learner config.
2. Log mean/std of the advantages over one rollout to confirm the bias.
3. Depending on 1-2: align the reward settings, centre the advantages, or
   lengthen `value_burnin_epochs` (currently 8).
