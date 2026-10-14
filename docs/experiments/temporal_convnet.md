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
* All runs were data-bound (~0.25 s/step data vs ~0.04 s/step model, even
  with 8 workers), so wall-clock time does not compare architectures.

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
