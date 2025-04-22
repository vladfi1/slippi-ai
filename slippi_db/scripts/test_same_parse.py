from collections import Counter
from concurrent import futures
import os
import random
import time

from absl import app
from absl import flags
from absl import logging
import tqdm

from slippi_db import preprocessing
from slippi_db import utils

ROOT = flags.DEFINE_string('root', None, 'root directory', required=True)
MAX_FILES = flags.DEFINE_integer('max_files', None, 'max files to process')
THREADS = flags.DEFINE_integer('threads', 1, 'number of threads')
LOG_INTERVAL = flags.DEFINE_integer('log_interval', 20, 'log interval')

def test(file: utils.LocalFile, tmpdir: str) -> dict:
  with file.extract(tmpdir) as path:
    meta = preprocessing.get_metadata_safe(path)
    valid, reason = preprocessing.is_training_replay(meta)
    if not valid:
      return {'outcome': 'skipped', 'reason': reason}

    try:
      preprocessing.assert_same_parse(path)
    except AssertionError as e:
      return {'outcome': 'failed', 'reason': e.args[0]}

    return {'outcome': 'passed'}

def main(_):
  root = ROOT.value
  if os.path.isdir(root):
    files = utils.traverse_slp_files(root)
  elif root.endswith('.7z'):
    logging.warning('7z files are very slow to traverse')
    files = utils.traverse_slp_files_7z(root)
  elif root.endswith('.zip'):
    files = utils.traverse_slp_files_zip(root)
  else:
    raise ValueError(f'Unknown file type: {root}')

  print(f'Found {len(files)} files.')
  if MAX_FILES.value is not None and len(files) > MAX_FILES.value:
    files = random.sample(files, MAX_FILES.value)
  print(f'Processing {len(files)} files.')

  tmpdir = utils.get_tmp_dir(in_memory=True)

  if THREADS.value == 1:
    results = [test(file, tmpdir) for file in tqdm.tqdm(files)]
  else:
    with futures.ProcessPoolExecutor(max_workers=THREADS.value) as executor:
      results = []
      todo = [executor.submit(test, file, tmpdir) for file in files]
      as_completed = futures.as_completed(todo)
      last_log = time.perf_counter() - LOG_INTERVAL.value + 5
      for f in tqdm.tqdm(as_completed, total=len(files)):
        results.append(f.result())

        now = time.perf_counter()
        if now - last_log > LOG_INTERVAL.value:
          last_log = now
          outcomes = Counter(r['outcome'] for r in results)
          print(outcomes)
          if outcomes['failed'] > 0:
            print(Counter(filter(None, [r.get('reason') for r in results])))

  outcomes = Counter(r['outcome'] for r in results)
  print("Outcomes", outcomes)
  print("Reasons", Counter(filter(None, [r.get('reason') for r in results])))

  if outcomes['failed'] > 0:
    return 1

if __name__ == '__main__':
  app.run(main)
