import multiprocessing as mp
import os
import time

from absl import app
from absl import flags
import numpy as np
import tree

from slippi_ai import eval_lib, dolphin, utils
from slippi_ai.data import CompressedGame

flags.DEFINE_string('dolphin_path', None, 'Path to dolphin directory.', required=True)
flags.DEFINE_string('iso_path', None, 'Path to SSBM iso.', required=True)
flags.DEFINE_string('tag', None, 'Experiment tag to pull from S3.')
flags.DEFINE_string('saved_model', None, 'Path to local saved model.')
flags.DEFINE_integer('batch_size', None, 'batch size', required=True)
flags.DEFINE_integer('runtime', 10, 'How long to run, in seconds.')
flags.DEFINE_boolean('render', False, 'render graphics')
flags.DEFINE_integer('delay', 0, 'number of frames of delay')

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main(_):
  players = {1: dolphin.AI(), 2: dolphin.CPU()}
  envs = []

  for i in range(FLAGS.batch_size):
    envs.append(eval_lib.AsyncEnv(
        dolphin_path=FLAGS.dolphin_path,
        iso_path=FLAGS.iso_path,
        players=players,
        slippi_port=51441 + i,
        render=FLAGS.render,
        exe_name='dolphin-emu-nogui',
        env_vars=dict(vblank_mode='0'),
    ))

  if FLAGS.saved_model:
    policy = eval_lib.Policy.from_saved_model(FLAGS.saved_model)
  elif FLAGS.tag:
    policy = eval_lib.Policy.from_experiment(FLAGS.tag)
  else:
    assert False

  hidden_state = policy.initial_state(FLAGS.batch_size)
  counts = np.full([FLAGS.batch_size], 0)
  rewards = np.full([FLAGS.batch_size], 0, dtype=np.float32)

  for _ in range(FLAGS.delay + 1):
    for env in envs:
      env.send({})

  def step(hidden_state):
    gamestates = [env.recv() for env in envs]
    batched_gamestates = utils.batch_nest(gamestates)[1]
    compressed_game = CompressedGame(batched_gamestates, counts, rewards)

    sampled_controller, hidden_state = policy.sample(
        compressed_game, hidden_state)
    sampled_controller = tree.map_structure(lambda x: x.numpy(), sampled_controller)

    for i, env in enumerate(envs):
      controller = tree.map_structure(lambda x: x[i], sampled_controller)
      env.send({1: controller})
      # env.send({})

    return hidden_state

  # warmup
  hidden_state = step(hidden_state)

  start_time = time.perf_counter()
  runtime = 0
  num_iters = 0

  while runtime < FLAGS.runtime:
    hidden_state = step(hidden_state)
    num_iters += 1
    runtime = time.perf_counter() - start_time

  for env in envs:
    env.stop()

  sps = num_iters / runtime
  total_sps = sps * FLAGS.batch_size

  print(f'sps: {sps:.0f}, total_sps: {total_sps:.0f}')

if __name__ == '__main__':
  mp.set_start_method('spawn')
  # del os.environ['CUDA_VISIBLE_DEVICES']
  app.run(main)
