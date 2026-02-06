# create a tiny network and run for 10 seconds on demo data

DATASET=slippi_ai/data/toy_dataset

python scripts/train.py \
  --config.dataset.data_dir=$DATASET/games \
  --config.dataset.meta_path=$DATASET/meta.json \
  --config.dataset.test_ratio=0.5 \
  --config.data.compressed=True \
  --config.data.balance_characters=True \
  --config.data.batch_size=2 \
  --config.data.num_workers=0 \
  --config.learner.minibatch_size=1 \
  --config.runtime.log_interval=4 \
  --config.runtime.max_runtime=10 \
  --config.runtime.num_evals_per_epoch=1 \
  --config.runtime.max_eval_steps=2 \
  --config.network.name=tx_like \
  --config.network.tx_like.hidden_size=2 \
  --config.value_function.separate_network_config=False \
  --config.controller_head.name=autoregressive \
  --config.controller_head.autoregressive.residual_size=2 \
  "$@"
