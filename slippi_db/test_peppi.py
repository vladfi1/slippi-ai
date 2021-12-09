import os
from typing import Tuple

from melee import enums, Character

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

BANNED_CHARACTERS = set([
    Character.KIRBY.value,
])

MIN_TIME = 60 * 60  # one minute

def is_training_replay(meta_dict: dict) -> bool:
  if meta_dict.get('invalid', False):
    return False

  del meta_dict['_id']
  meta = preprocessing.Metadata.from_dict(meta_dict)

  if meta.num_players != 2:
    return False
  
  for player in meta.players:
    if player.type != 0:
      return False
    
    if player.character in BANNED_CHARACTERS:
      return False
  
  if meta.lastFrame < MIN_TIME:
    return False
  
  if enums.to_internal_stage(meta.stage) == enums.Stage.NO_STAGE:
    return False
  
  return True

def get_singles_info(env: str):
  meta_db = upload_lib.get_db(env, 'meta')
  metas = meta_db.find()
  metas = filter(is_training_replay, metas)
  return list(metas)
