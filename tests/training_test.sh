# create a tiny network and run for 10 seconds on demo data

DATASET=slippi_ai/data/toy_dataset

python scripts/train.py \
  --config.dataset.data_dir=$DATASET/games \
  --config.dataset.meta_path=$DATASET/meta.json \
  --config.dataset.test_ratio=0.5 \
  --config.data.compressed=True \
  --config.runtime.log_interval=4 \
  --config.runtime.max_runtime=10 \
  --config.runtime.eval_every_n=2 \
  --config.runtime.num_eval_steps=1 \
  --config.network.mlp.depth=1 \
  --config.network.mlp.width=1 \
  --config.value_function.separate_network_config=False \
  "$@"
