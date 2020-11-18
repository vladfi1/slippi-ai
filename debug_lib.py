'''
   Load a trained model, evaluate it on test set, 
   writing test replay data and model predictions
   to a human readable dataframe for data analysis.

   A mixture of classes from multiple scripts used by debug_test.py
'''

import sys
import datetime
import embed
import os
import secrets
import sys

import tree
import utils

import numpy as np
import sonnet as snt
import tensorflow as tf
import pandas as pd

import data
import networks
import paths
import stats
import utils

from train_lib import sanitize_batch, sanitize_game
from learner import to_time_major
from policies import Policy

from collections import defaultdict, namedtuple

class DebugManager:
  '''
    Creates and writes pd.DataFrame tables of observed controller states,
    model predicted controller states, model sampled controller states, and loss

    Related to train_lib.TrainManager
  '''

  DEFAULT_CONFIG = dict(
      table_length=5e4,
      table_name='debug_test',
  )

  def __init__(self, 
      learner, 
      data_source, 
      saved_model_path, 
      table_length,
      table_name,
    ):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.saved_model_path = saved_model_path

    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

    self.mdf = pd.DataFrame()
    self.time = 0
    self.table_length = table_length
    self.table_name = table_name

  def step(self):
    with self.data_profiler:
      batch = sanitize_batch(next(self.data_source))
    with self.step_profiler:
      # debug_namedtuple = self.learner.step(batch, self.hidden_state)
      debug_namedtuple = self.learner.compiled_step(batch, self.hidden_state)
      self.hidden_state = debug_namedtuple.final_state
    self.update_table(debug_namedtuple)
    return debug_namedtuple.loss

  def update_table(self, debug_namedtuple):
    nests = {
      'obs': debug_namedtuple.observations, 
      'sample': debug_namedtuple.samples,
      'gamestate': debug_namedtuple.gamestate,
      'dist': debug_namedtuple.distances
    }
    path_delim = '.'

    df = None
    for nest_name, nest in nests.items():

      # E.g., skip dist for controller_heads that do not yet provide controller component-wise distance
      if nest_name == 'dist' and not tf.nest.is_nested(nest):
        continue

      for path, array in tree.flatten_with_path(nest):
        leaf_name = path_delim.join([str(subpath) for subpath in path])

        wide_single_df = pd.DataFrame(tf.squeeze(array).numpy().astype(float))
        wide_single_df['Time'] = wide_single_df.index
        col_nm = f'{nest_name} {leaf_name}'
        long_df = pd.melt(wide_single_df, id_vars=['Time'],
          var_name='Batch', value_name=col_nm)

        if df is not None:
          df[col_nm] = long_df[col_nm]
        else:
          df = long_df

    df['Time'] += self.time
    self.time += len(wide_single_df)
    df['Loss'] = df[[col for col in df.columns if 'dist ' in col]].agg(np.nansum, axis='columns')

    self.mdf = self.mdf.append(df)

    if len(self.mdf) > self.table_length:
      print(f'Reached or exceeded specified debug.max_table_size: {self.table_length}')
      out_fn = self.saved_model_path + f'{self.table_name}.csv'
      self.mdf.to_csv(out_fn)
      print(f'Wrote {len(self.mdf)} frames to {out_fn}')
      print(f'Exiting.')
      sys.exit(0)
    return


class DebugLearner:
  '''
    Related to learner.Learner
  '''

  DEFAULT_CONFIG = dict(
  )

  def __init__(self,
      policy: Policy):
    self.policy = policy
    self.compiled_step = tf.function(self.step)
    self.sample = lambda *structs: policy.sample(*tf.nest.flatten(structs))

  def step(self, batch, initial_states):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.policy.initial_state(restarting.shape[0]),
        initial_states)

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    p1_controller = tm_gamestate['player'][1]['controller_state']
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    next_action_distances, final_state = self.policy.loss(tm_gamestate, initial_states)
    mean_distances = tf.nest.map_structure(
        tf.reduce_mean, next_action_distances)
    loss = tf.add_n(tf.nest.flatten(mean_distances))

    controller_samples, _ = utils.dynamic_rnn(self.sample, tm_gamestate, initial_states)

    debug_res = namedtuple('debug_res', [
      'loss',
      'observations',
      'samples',
      'distances',
      'gamestate',
      'final_state'
    ])
    return debug_res(
      loss, 
      next_action,
      controller_samples, 
      next_action_distances, 
      tm_gamestate,
      final_state
    )
