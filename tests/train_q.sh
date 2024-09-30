# create a tiny network and run for 10 seconds on demo data

DATASET=slippi_ai/data/toy_dataset

python scripts/train_q.py \
  --config.dataset.data_dir=$DATASET/games \
  --config.dataset.meta_path=$DATASET/meta.json \
  --config.dataset.test_ratio=0.5 \
  --config.data.compressed=True \
  --config.data.batch_size=2 \
  --config.data.unroll_length=2 \
  --config.runtime.log_interval=4 \
  --config.runtime.max_runtime=10 \
  --config.runtime.eval_every_n=2 \
  --config.runtime.num_eval_steps=1 \
  --config.network.name=gru \
  --config.network.gru.hidden_size=3 \
  --config.q_function.network.name=gru \
  --config.q_function.network.gru.hidden_size=3 \
  "$@"
