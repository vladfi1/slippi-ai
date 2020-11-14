"""Learner test script."""

import os

from absl import app
from absl import flags

import numpy as np
import sonnet as snt
import tensorflow as tf

import melee
import embed

REPLAY_PATH = os.path.expanduser('~/Slippi/Game_20201113T162336.slp')

def read_gamestates(replay_path):
  console = melee.Console(is_dolphin=False,
                          allow_old_version=False,
                          path=replay_path)
  console.connect()

  while True:
    gamestate = console.step()
    if gamestate is None:
      break
    yield gamestate

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

class Learner:

  def __init__(self, embed_game):
    self.network = snt.nets.MLP([256, 128, embed.embed_controller.size])
    self.optimizer = snt.optimizers.Adam(1e-4)
    self.embed_game = embed_game

  def step(self, batch_gamestate):
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
  gamestates = list(read_gamestates(REPLAY_PATH))
  print(len(gamestates))

  embed_game = embed.make_game_embedding()
  trajectory = make_trajectory(embed_game, gamestates[:10])

  trajectories = [trajectory] * 32
  batch_trajectories = batch_nest(trajectories)

  # batch_signature = tf.nest.map_structure(
  #     lambda a: (a.dtype, a.shape),
  #     batch_trajectories)
  # print(batch_signature)

  learner = Learner(embed_game)

  for _ in range(1000):
    loss = learner.step(batch_trajectories)
    print(loss.numpy())

if __name__ == '__main__':
  app.run(main)
