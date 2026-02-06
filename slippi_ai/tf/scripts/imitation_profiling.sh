#!/usr/bin/env sh

# This example script trains a 3-layer transformer-like model on Fox data with
# a delay of 18 frames. You should set the batch_size to the highest possible
# value that fits in your GPU memory.

DATA_ROOT=slippi_ai/data/toy_dataset/
DATA_DIR="$DATA_ROOT/games"
META_PATH="$DATA_ROOT/meta.json"

python3 scripts/train.py \
  --wandb.mode=disabled \
  --config.tag=all_delay_18 \
  --config.policy.delay=18 \
  --config.data.batch_size=512 \
  --config.data.unroll_length=80 \
  --config.data.cached \
  --config.learner.learning_rate=1e-4 \
  --config.learner.reward_halflife=4 \
  --config.network.name=tx_like \
  --config.network.tx_like.num_layers=3 \
  --config.network.tx_like.hidden_size=512 \
  --config.network.tx_like.ffw_multiplier=2 \
  --config.policy.train_value_head=False \
  --config.value_function.train_separate_network=True \
  --config.value_function.separate_network_config=True \
  --config.value_function.network.name=tx_like \
  --config.value_function.network.tx_like.num_layers=1 \
  --config.value_function.network.tx_like.hidden_size=512 \
  --config.value_function.network.tx_like.ffw_multiplier=2 \
  --config.controller_head.name=autoregressive \
  --config.controller_head.autoregressive.component_depth=2 \
  --config.controller_head.autoregressive.residual_size=128 \
  --config.dataset.allowed_characters=all \
  --config.dataset.allowed_opponents=all \
  --config.dataset.data_dir=$DATA_DIR \
  --config.dataset.meta_path=$META_PATH \
  --config.dataset.test_ratio=0.5 \
  --config.runtime.eval_every_n=10000 \
  --config.runtime.max_runtime=30 \
  --config.runtime.log_interval=10 \
  --config.runtime.save_interval=600 \
  "$@"
