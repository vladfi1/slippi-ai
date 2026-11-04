# Temporal convolution networks vs tx_like (2026-10-14)

Goal: match `falco_d21_tx3x1024_rating` (tx_like, 3 layers x 1024, LSTM
memory) at 0.5 epochs with the `tcn` network
(`slippi_ai/jax/scripts/imitation_tcn.py`).

All runs: falco, delay 21, with rating, hidden size 1024, batch 1024,
`runtime.max_epochs=0.5`, wandb project `slippi-ai`. Metric is the full
test-set `eval.policy.loss`, measured every 0.25 epochs. Learning rate is 1e-4
unless noted.

## Results

| wandb run | network | net params | eval @0.25 | eval @0.5 |
|---|---|---|---|---|
| `falco_d21_tx3x1024_rating` (reference, 2026-07-29) | tx_like 3x1024 | ~44M | 0.846 | **0.811** |
| `falco_d21_tcn3x1024m5_rating` (batch 512) | linear block, 3 layers, m5 | | 0.877 (@0.22) | 0.860 |
| `falco_d21_tcn3x1024m5_gated_tanh_sigmoid_rating` | gated block, 3 layers, m5 | 59.7M | 0.862 | killed |
| `falco_d21_tcn8x1024m3b2_depthwise_rating` | depthwise block, 8 layers, m3, base 2 | 65.0M | 0.852 | **0.822** |
| `falco_d21_tcn8x1024m1b2_gated_tanh_sigmoid_rating` | gated block, 8 layers, m1, base 2 | 81.8M | 0.855 | killed |
| `falco_d21_tcn8x1024m3b2_depthwise_lr3e-4_rating` | depthwise, 8 layers, m3, base 2, **LR 3e-4** | 65.0M | 0.834 | 0.806 |

Notes:

* Param counts are the policy network only; the controller head is 0.88M in
  every run.
* The depthwise LR 1e-4 run's 0.5 eval comes from resuming its 0.25
  checkpoint, because the first attempt stopped at `max_epochs` just before
  the scheduled eval (fixed in `train_policy.py`, not yet in `train_lib.py`).
* The LR 3e-4 run beats tx_like, but tx_like was only run at 1e-4, so that
  comparison is confounded. We are keeping LR fixed at 1e-4 for architecture
  comparisons; at 1e-4 the best TCN is still ~0.011 behind tx_like at 0.5.
* The logged `timing: data=... step=...` split does not show whether a run is
  data- or compute-bound: `step` only times the async dispatch, and waiting on
  the GPU shows up under `data`. The TCN runs ran at ~4 steps/s; whether they
  were data-bound was not verified (e.g. with GPU utilization).

## What changed in the network

The original `linear` block applies a dense dilated conv straight to the
residual stream: no pre-norm, non-zero init, purely linear time mixing, and
with the default `dilation_base = memory_size + 1` the receptive fields tile
without overlap.

* `block='gated'`: pre-norm residual branch, LayerNorm -> dense dilated conv
  to 2h -> gate (`tanh_sigmoid` as in WaveNet, or `glu`) -> zero-initialized
  linear. Recovered about half the gap at 0.25 epochs.
* `block='depthwise'`: LayerNorm -> depthwise dilated conv (each channel gets
  its own `memory_size + 1` taps, no cross-channel mixing) -> Linear h->2h ->
  gate -> zero-initialized linear. Time mixing is nearly free, so the budget
  goes into depth: 8 layers with `dilation_base=2` (dilations 1..128, kernel
  4) see ~766 frames back. Spending the same budget on dense convs (8 layers,
  m1) was no better.
* LR 3e-4 was the largest single gain (~0.016 at 0.5 epochs); the pre-norm,
  zero-init stack is stable there.

Best config at LR 1e-4:

```
python slippi_ai/jax/scripts/imitation_tcn.py \
  --net.block depthwise --net.memory_size 3 --net.dilation_base 2 \
  --config.network.tcn.num_layers=8
```

## References

WaveNet (gated dilated convs), Bai et al. 2018 "An Empirical Evaluation of
Generic Convolutional and Recurrent Networks", ByteNet, ConvS2S (GLU), and
the short depthwise convs in Mamba / Hyena / ConvNeXt.

# Mamba2 vs tx_like (2026-10-14)

Same setup as above (LR 1e-4, batch 1024, 0.5 epochs), with the `mamba2`
network (`slippi_ai/jax/scripts/imitation_mamba2.py`): 4 layers of pre-norm
residual Mamba2 mixer (from
[mamba2-jax](https://github.com/jax-state-spaces/mamba2-jax), expand 2,
state size 128, head dim 64, conv kernel 4) each followed by a FFW ResBlock
with multiplier 2, as in tx_like.

| wandb run | net params | eval @15.3k steps | eval @30.7k steps | steps/s |
|---|---|---|---|---|
| `falco_d21_tx3x1024_rating` (July reference, older dataset, unroll 80) | 44.0M | 0.846 | 0.811 | |
| `falco_d21_tx3x1024_rating_ds318` | 44.0M | 0.845 | 0.813 | 5.2 |
| `falco_d21_tx4x1024_rating` | 56.6M | 0.841 | 0.806 | 4.0 |
| `falco_d21_tcn8x1024m3b2_depthwise_rating` | 65.0M | 0.852 | 0.822 | |
| `falco_d21_mamba3x1024s128_rating` | 38.6M | 0.839 | 0.806 | 3.2 |
| `falco_d21_mamba4x1024s128_rating` | 49.4M | **0.834** | **0.800** | 2.5 |

Evals are at the same training steps in every run (~0.25 / ~0.5 epochs; the
logged epoch varies slightly between runs). Except for the July reference, all
runs use `Dataset-3.18.0` with unroll 84; re-running tx_like 3x1024 on it gave
the same result, so the dataset change does not matter here.

Train loss (mean per 4k steps), first bin omitted:

| run | | | | | | | |
|---|---|---|---|---|---|---|---|
| tx_like 3x1024 (ds318) | 0.885 | 0.866 | 0.847 | 0.823 | 0.824 | 0.814 | 0.817 |
| tx_like 4x1024 | 0.897 | 0.862 | 0.851 | 0.832 | 0.817 | 0.815 | 0.812 |
| Mamba2 3x1024 | 0.885 | 0.857 | 0.846 | 0.831 | 0.831 | 0.809 | 0.807 |
| Mamba2 4x1024 | 0.887 | 0.847 | 0.841 | 0.827 | 0.808 | 0.799 | 0.799 |

All net params include the embedding, as logged. Per layer at h=1024 a
Mamba2 mixer is ~6.6M (in_proj 1024x4384, out_proj 2048x1024) vs ~8.4M for an
LSTM, plus ~4.2M for the FFW in both.

Conclusions (single seeds, so ~0.005 differences may be noise):

* At equal depth Mamba2 beats tx_like by ~0.006 with fewer params (3 layers:
  0.806 vs 0.813; 4 layers: 0.800 vs 0.806), so the gain is not only from
  the extra layer that its smaller layers allow.
* Going from 3 to 4 layers helps both by ~0.006.
* Mamba2 3x1024 matches tx_like 4x1024 with 18M fewer params, but runs at
  3.2 vs 4.0 steps/s; per unit of wall-clock time tx_like is about even or
  ahead.

Implementation notes:

* The parameters and projections are the library's `Mamba2Mixer`, but the
  SSD scan is our own: the library's broadcasts B/C to every head and its
  einsums materialize multi-GB intermediates (67GB at batch 1024 x 84). Ours
  matches the library to 1e-4 (`test_mamba2_matches_library`).
* With `chunk_size >= unroll_length` the SSD runs as a single chunk, which was
  2x faster and half the memory of chunk 32. The SSD is checkpointed.
* Projections run in bf16; the conv, SSD and gated norm run in fp32.
* Resets in the middle of an unroll are honored, matching step: terms that
  cross a reset are masked out of the SSD and the conv. (Earlier runs ignored
  them, which broke RL, where sim-env games end mid-rollout.)
* Mamba2 4x1024 is compute-bound (GPU utilization 89-100% over a few-minute check):
  2.5 steps/s (~0.4 s/step), about 3.7 hours for 0.5
  epochs. The SSD is ~30 ms of the ~80 ms per layer (fwd + bwd). 3x1024 ran
  at 3.2 steps/s.
