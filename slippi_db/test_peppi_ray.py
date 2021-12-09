"""Tests that peppi and libmelee agree on a subset of files.

Stores results in the 'peppi_test' db.

Run remotely with:

ray submit --start slippi_db/parsing_cluster.yaml \
    slippi_db/test_peppi_ray.py \
    --env test --num 10
"""

import random
import time

from absl import app
from absl import flags
import pymongo
import ray

from slippi_db import test_peppi
from slippi_db.upload_lib import get_db

flags.DEFINE_string('env', 'test', 'production environment')
flags.DEFINE_integer('num', 10, 'number of files to test')
flags.DEFINE_boolean('wipe', False, 'wipe previous tests')
flags.DEFINE_boolean('local', False, 'run locally')

FLAGS = flags.FLAGS

check_slp = ray.remote(num_cpus=1)(test_peppi.check_slp)

TEST_KEY = 'peppi_test'

def check_multiple(env: str, num: int):
  start_time = time.perf_counter()
  singles_infos = test_peppi.get_singles_info(env)

  to_test = random.choices(singles_infos, k=num)
  keys = [d['key'] for d in to_test]

  results = []

  for key in keys:
    results.append(check_slp.remote(env, key))
  results = ray.get(results)

  test_db = get_db(env, TEST_KEY)
  if FLAGS.wipe:
    test_db.delete_many({})

  updates = []
  num_passed = 0
  for key, (passed, message) in zip(keys, results):
    num_passed += int(passed)
    peppi_test = {'passed': passed, 'message': message, 'key': key}
    updates.append(pymongo.UpdateOne(
        {'key': key}, {'$set': peppi_test}, upsert=True))
  test_db.bulk_write(updates)

  run_time = time.perf_counter() - start_time
  fps = num / run_time
  print(f'num: {num}, time: {run_time:.1f}, fps: {fps:.1f}, passed: {num_passed}')

def main(_):
  if not FLAGS.local:
    ray.init('auto')
  check_multiple(FLAGS.env, FLAGS.num)

if __name__ == '__main__':
  app.run(main)
