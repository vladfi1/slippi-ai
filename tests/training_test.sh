# create a tiny network and run for 10 seconds on demo data

# don't write to mongodb by accident
MONGO_URI=

python scripts/train.py with \
  dataset.data_dir=data/pq/ \
  dataset.test_ratio=0.5 \
  data.compressed=False \
  runtime.log_interval=4 \
  runtime.max_runtime=10 \
  evaluation.eval_every_n=2 \
  evaluation.num_eval_steps=1 \
  network.mlp.depth=1 \
  network.mlp.width=1
