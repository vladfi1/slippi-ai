"""Learner test script."""

import itertools
import os

from absl import app
from absl import flags

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee
import embed
import stats

REPLAY_PATH = os.path.expanduser('~/Slippi/Game_20201113T162336.slp')
DATASET_PATH = '/media/vlad/Archive-2T-Nomad/Vlad/training_data/'

def read_gamestates(replay_path):
  print("Reading from ", replay_path)
  console = melee.Console(is_dolphin=False,
                          allow_old_version=True,
                          path=replay_path)
  console.connect()

  gamestate = console.step()
  port_map = dict(zip(gamestate.player.keys(), [1, 2]))

  def fix_state(s):
    s.player = {port_map[p]: v for p, v in s.player.items()}

  while gamestate:
    fix_state(gamestate)
    yield gamestate
    gamestate = console.step()

class TrajectoryProducer:
  def __init__(self, embed_game):
    self.embed_game = embed_game

  def __iter__(self):
    for name in itertools.cycle(stats.table['filename']):
      path = DATASET_PATH + name
      yield [self.embed_game.from_state(s) for s in read_gamestates(path)]
      # TODO: also yield with players swapped?

def np_array(*vals):
  return np.array(vals)

def batch_nest(nests):
  return tf.nest.map_structure(np_array, *nests)

def make_trajectory(embed_game, states):
  return batch_nest(embed_game.from_state(s) for s in states)

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class TrajectoryManager:
  # TODO: manage recurrent state? can also do it in the learner

  def __init__(self, source, num_frames: int):
    self.source = source
    self.num_frames = num_frames
    self.frames = self.iter_frames()

  def iter_frames(self):
    for trajectory in self.source:
      restart = True
      for frame in trajectory:
        yield frame, restart
        restart = False

  def next(self):
    frames, restarts = zip(*[next(self.frames) for _ in range(self.num_frames)])
    return batch_nest(frames), np.array(restarts)

class DataSource:
  def __init__(self, embed_game, batch_size=64, unroll_length=64):
    producer = iter(TrajectoryProducer(embed_game))
    self.managers = [
        TrajectoryManager(producer, unroll_length)
        for _ in range(batch_size)]

  def __next__(self):
    return batch_nest([m.next() for m in self.managers])

class Learner:

  def __init__(self, embed_game):
    self.network = snt.nets.MLP([256, 128, embed.embed_controller.size])
    self.optimizer = snt.optimizers.Adam(1e-4)
    self.embed_game = embed_game

  def step(self, batch_gamestate, restarting):
    time_major_gamestate = tf.nest.map_structure(to_time_major, batch_gamestate)

    flat_gamestate = self.embed_game(time_major_gamestate)
    prev_gamestate = flat_gamestate[:-1]

    p1_controller = time_major_gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    with tf.GradientTape() as tape:
      outputs = self.network(prev_gamestate)
      next_action_distances = embed.embed_controller.distance(
          outputs, next_action)
      mean_distances = tf.nest.map_structure(
          tf.reduce_mean, next_action_distances)
      loss = tf.add_n(tf.nest.flatten(mean_distances))

    params = self.network.trainable_variables
    grads = tape.gradient(loss, params)
    self.optimizer.apply(grads, params)
    return loss

def main(_):
  embed_game = embed.make_game_embedding()
  learner = Learner(embed_game)
  data_source = DataSource(embed_game)

  for _ in range(1000):
    loss = learner.step(*next(data_source))
    print(loss.numpy())

if __name__ == '__main__':
  app.run(main)
