import os
from typing import Tuple

from slippi_ai import types
from slippi_db import preprocessing
from slippi_db import upload_lib

def check_slp(env: str, key: str) -> Tuple[bool, str]:
  slp_file = upload_lib.download_slp_to_file(env, key)

  message = None
  try:
    preprocessing.assert_same_parse(slp_file.name)
    passed = True
  except types.InvalidGameError as e:
    message = f'InvalidGameError: {e}'
    passed = True
  except AssertionError as e:
    message = 'AssertionError: ' + str(e)
    passed = False
  except Exception as e:
    message = f'{type(e)}: {e}'
    passed = False
  finally:
    os.remove(slp_file.name)

  return passed, message

def get_singles_info(env: str):
  meta_db = upload_lib.get_db(env, 'meta')
  metas = meta_db.find()
  metas = filter(preprocessing.is_training_replay, metas)
  return list(metas)
