"""Model state cache and user preferences, persisted across restarts."""

import dataclasses
import json
import logging
import os
import pickle
import typing as tp
from typing import Optional

from slippi_ai import saving
from slippi_ai import eval_lib

@dataclasses.dataclass
class LoadedModel:
  mtime: float  # file modification time when loaded
  size: int  # file size when loaded
  state: dict  # trimmed to the keys shown by !config
  summary: eval_lib.AgentSummary

# The parts of a model's state that we keep in memory (and cache on disk).
# Everything else (e.g. the network weights) is only loaded in the sessions.
MODEL_STATE_KEYS = [
    'step', 'config', 'rl_config', 'agent_config', 'opponent', 'opponents',
]

# Bump this whenever the format of the cache entries changes.
MODEL_CACHE_VERSION = 1

def trim_model_state(state: dict) -> dict:
  return {k: state[k] for k in MODEL_STATE_KEYS if k in state}

class ModelCache:
  """On-disk cache of trimmed model states, one pickle file per model.

  An entry is only valid if the model file's mtime and size match those
  recorded when the entry was written.
  """

  def __init__(self, path: str):
    self._path = path
    os.makedirs(path, exist_ok=True)

  def _entry_path(self, model: str) -> str:
    return os.path.join(self._path, model)

  def get(self, model: str, mtime: float, size: int) -> Optional[dict]:
    """Returns the cached trimmed state, or None if missing or stale."""
    entry_path = self._entry_path(model)
    if not os.path.isfile(entry_path):
      return None

    try:
      with open(entry_path, 'rb') as f:
        entry = saving.CustomUnpickler(f).load()
    except Exception as e:
      logging.warning(f'Failed to read model cache entry {entry_path}: {e}')
      return None

    if (
        not isinstance(entry, dict)
        or entry.get('version') != MODEL_CACHE_VERSION
        or entry.get('mtime') != mtime
        or entry.get('size') != size
        or set(entry.get('keys', [])) != set(MODEL_STATE_KEYS)
    ):
      return None

    return entry['state']

  def put(self, model: str, mtime: float, size: int, state: dict):
    entry = dict(
        version=MODEL_CACHE_VERSION,
        mtime=mtime,
        size=size,
        keys=MODEL_STATE_KEYS,
        state=state,
    )
    entry_path = self._entry_path(model)
    # Write to a temporary file and rename so that a concurrent reader (or a
    # crash mid-write) never sees a partial entry.
    tmp_path = f'{entry_path}.tmp{os.getpid()}'
    try:
      with open(tmp_path, 'wb') as f:
        pickle.dump(entry, f)
      os.replace(tmp_path, entry_path)
    except Exception as e:
      logging.warning(f'Failed to write model cache entry {entry_path}: {e}')
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  def prune(self, current_models: tp.Collection[str]):
    """Removes entries for models that no longer exist."""
    for entry in os.listdir(self._path):
      if entry not in current_models:
        entry_path = self._entry_path(entry)
        if os.path.isfile(entry_path):
          logging.info(f'Removing stale model cache entry {entry}')
          try:
            os.remove(entry_path)
          except FileNotFoundError:
            pass  # Another process may have removed it.

class UserPrefs:
  """Per-user preferences persisted to a JSON file.

  The file maps user id (e.g. "twitch:<name>") to a dict of preferences.
  """

  def __init__(self, path: Optional[str]):
    self._path = path
    self._prefs: dict[str, dict[str, tp.Any]] = {}
    self._load()

  def _load(self):
    if self._path is None or not os.path.isfile(self._path):
      return
    try:
      with open(self._path) as f:
        prefs = json.load(f)
    except Exception as e:
      logging.warning(f'Failed to read user prefs from {self._path}: {e}')
      return
    if not isinstance(prefs, dict):
      logging.warning(f'Ignoring malformed user prefs in {self._path}')
      return
    self._prefs = prefs
    logging.info(f'Loaded prefs for {len(prefs)} users from {self._path}')

  def _save(self):
    if self._path is None:
      return
    os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
    tmp_path = f'{self._path}.tmp{os.getpid()}'
    try:
      with open(tmp_path, 'w') as f:
        json.dump(self._prefs, f, indent=2, sort_keys=True)
      os.replace(tmp_path, self._path)
    except Exception as e:
      logging.warning(f'Failed to write user prefs to {self._path}: {e}')
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  def get(self, user: str, key: str) -> tp.Any:
    return self._prefs.get(user, {}).get(key)

  def set(self, user: str, key: str, value: tp.Any):
    self._prefs.setdefault(user, {})[key] = value
    self._save()

  def rename_users(self, rename: tp.Callable[[str], str]):
    """Renames users, e.g. to migrate to a new format of user ids."""
    renamed = {rename(user): prefs for user, prefs in self._prefs.items()}
    if renamed.keys() != self._prefs.keys():
      self._prefs = renamed
      self._save()

  def delete(self, user: str, key: str):
    user_prefs = self._prefs.get(user)
    if user_prefs is None or key not in user_prefs:
      return
    del user_prefs[key]
    if not user_prefs:
      del self._prefs[user]
    self._save()
