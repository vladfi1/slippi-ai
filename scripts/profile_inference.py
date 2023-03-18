import time

from absl import app
from absl import flags
import numpy as np
import tree

from slippi_ai import eval_lib, embed, data

flags.DEFINE_string('tag', None, 'Experiment tag to pull from S3.')
flags.DEFINE_string('saved_model', None, 'Path to local saved model.')
flags.DEFINE_integer('batch_size', None, 'batch size', required=True)
flags.DEFINE_integer('runtime', 5, 'How long to run, in seconds.')

FLAGS = flags.FLAGS

embed_game = embed.make_game_embedding()
dummy_game = embed_game.dummy()
dummy_compressed_game = data.CompressedGame(
    states=dummy_game,
    counts=0,
    rewards=0,
)

def main(_):
  if FLAGS.saved_model:
    policy = eval_lib.Policy.from_saved_model(FLAGS.saved_model)
  elif FLAGS.tag:
    policy = eval_lib.Policy.from_experiment(FLAGS.tag)
  else:
    assert False

  compressed_game = tree.map_structure(
      lambda x: np.full([FLAGS.batch_size], x),
      dummy_compressed_game,
  )

  hidden_state = policy.initial_state(FLAGS.batch_size)

  # warmup
  policy.sample(compressed_game, hidden_state)

  start_time = time.perf_counter()
  runtime = 0
  num_iters = 0

  while runtime < FLAGS.runtime:
    _, hidden_state = policy.sample(compressed_game, hidden_state)
    num_iters += 1
    runtime = time.perf_counter() - start_time

  sps = num_iters / runtime
  total_sps = sps * FLAGS.batch_size

  print(f'sps: {sps:.0f}, total_sps: {total_sps:.0f}')

if __name__ == '__main__':
  app.run(main)
