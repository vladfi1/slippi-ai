"""Published models: an online index, downloaded and cached on demand.

The index is a JSON file hosted next to the model files, so newly published
models show up without a new release:

  {
    "updated": "2026-10-09T12:00:00Z",
    "models": [
      {
        "name": "fox-d21",
        "description": "...",              # optional
        "url": "https://huggingface.co/<repo>/resolve/<commit>/fox-d21.onnx",
        "sha256": "...",
        "size": 98041677,                  # bytes
        "format_version": 1,               # onnx_policies.FORMAT_VERSION
        "batch_size": 1,
        "agent_type": "RL",                # eval_lib.AgentType name
        "characters": ["FOX"],             # melee.Character names
        "opponents": ["FALCO", "FOX"],
        "delay": 21,
        "published": "2026-10-09"          # optional
      }
    ]
  }

Unknown fields are ignored. An incompatible schema change gets a new file
name (index-v2.json), so older installs keep reading theirs. Each entry's URL
is pinned to the revision that uploaded it, and downloads are checked against
its sha256. Models whose format_version this install can't load are kept in
the index but aren't compatible().

The last fetched index is saved, so the list works offline, and downloads
are cached by hash along with their index entry, so a model removed from the
index stays playable once downloaded.
"""

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import sys
import threading
import typing as tp
import urllib.error
import urllib.request

INDEX_VERSION = 1
# TODO: Set once the models are hosted.
DEFAULT_INDEX_URL: tp.Optional[str] = None

INDEX_URL_ENV_VAR = 'PHILLIP_INDEX_URL'
CACHE_ENV_VAR = 'PHILLIP_CACHE'

# The GUI plays one game, and CUDA graphs need a fixed batch size.
PLAY_BATCH_SIZE = 1

TIMEOUT = 30  # seconds
USER_AGENT = 'phillip'
INFO_FILE = 'info.json'

# Called with (bytes downloaded, total bytes).
ProgressFn = tp.Callable[[int, int], None]


class DownloadCancelled(Exception):
  pass


@dataclasses.dataclass
class ModelInfo:
  name: str
  url: str
  sha256: str
  size: int
  format_version: int
  batch_size: tp.Optional[int]
  agent_type: str
  characters: list[str]
  opponents: list[str]
  delay: int
  description: str = ''
  published: str = ''

  @classmethod
  def from_json(cls, entry: dict) -> 'ModelInfo':
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in entry.items() if k in names})

  def to_json(self) -> dict:
    return dataclasses.asdict(self)

  @property
  def filename(self) -> str:
    return self.url.rsplit('/', 1)[-1].split('?', 1)[0]

  def compatible(self) -> bool:
    """Whether this install can play the model."""
    from slippi_ai import onnx_policies
    return (self.format_version == onnx_policies.FORMAT_VERSION
            and self.batch_size == PLAY_BATCH_SIZE)

  def summary(self):
    """An eval_lib.AgentSummary, like one read from the model file."""
    import melee
    from slippi_ai import eval_lib
    return eval_lib.AgentSummary(
        type=eval_lib.AgentType[self.agent_type],
        delay=self.delay,
        characters=[melee.Character[c] for c in self.characters],
        opponents=[melee.Character[c] for c in self.opponents],
    )


def sha256_file(path: str) -> str:
  sha256 = hashlib.sha256()
  with open(path, 'rb') as f:
    while chunk := f.read(1 << 20):
      sha256.update(chunk)
  return sha256.hexdigest()


def info_from_file(path: str, name: str, url: str, **kwargs) -> ModelInfo:
  """An index entry for an exported .onnx model, served at url."""
  from slippi_ai import eval_lib, onnx_policies
  state = onnx_policies.load_state_from_disk(path, load_model=False)
  metadata = state['onnx_metadata']
  summary = eval_lib.AgentSummary.from_state(state)
  return ModelInfo(
      name=name,
      url=url,
      sha256=sha256_file(path),
      size=os.path.getsize(path),
      format_version=metadata['format_version'],
      batch_size=metadata['batch_size'],
      agent_type=summary.type.name,
      characters=[c.name for c in summary.characters],
      opponents=[c.name for c in summary.opponents],
      delay=summary.delay,
      **kwargs,
  )


@dataclasses.dataclass
class Index:
  models: list[ModelInfo]
  updated: str = ''

  @classmethod
  def from_json(cls, index: dict) -> 'Index':
    models = []
    for entry in index['models']:
      try:
        models.append(ModelInfo.from_json(entry))
      except TypeError as e:  # Missing fields.
        logging.warning(f'Skipping index entry {entry.get("name")}: {e}')
    return cls(models=models, updated=index.get('updated', ''))

  def to_json(self) -> dict:
    return dict(updated=self.updated, models=[m.to_json() for m in self.models])


def cache_dir() -> pathlib.Path:
  """Where the index and downloaded models go; override with $PHILLIP_CACHE."""
  if CACHE_ENV_VAR in os.environ:
    return pathlib.Path(os.environ[CACHE_ENV_VAR])
  if sys.platform == 'win32':
    base = os.environ.get('LOCALAPPDATA') or pathlib.Path.home() / 'AppData' / 'Local'
  else:
    base = os.environ.get('XDG_CACHE_HOME') or pathlib.Path.home() / '.cache'
  return pathlib.Path(base) / 'phillip' / 'models'


def index_url() -> str:
  url = os.environ.get(INDEX_URL_ENV_VAR, DEFAULT_INDEX_URL)
  if not url:
    raise ValueError(
        f'No model index has been published yet; set ${INDEX_URL_ENV_VAR}.')
  return url


def saved_index_path() -> pathlib.Path:
  return cache_dir() / f'index-v{INDEX_VERSION}.json'


def _etag_path() -> pathlib.Path:
  return saved_index_path().with_suffix('.etag')


def _write_atomic(path: pathlib.Path, data: bytes):
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + '.tmp')
  tmp.write_bytes(data)
  os.replace(tmp, path)


def load_index() -> tp.Optional[Index]:
  """The last fetched index, or None if there isn't one."""
  path = saved_index_path()
  if not path.exists():
    return None
  try:
    return Index.from_json(json.loads(path.read_text(encoding='utf-8')))
  except (ValueError, KeyError) as e:
    logging.warning(f'Ignoring corrupt saved index {path}: {e}')
    return None


def fetch_index(url: tp.Optional[str] = None) -> Index:
  """Downloads the index and saves it; raises if it can't be fetched."""
  if url is None:
    url = index_url()
  request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
  saved = load_index()
  etag_path = _etag_path()
  if saved is not None and etag_path.exists():
    request.add_header('If-None-Match', etag_path.read_text().strip())

  try:
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
      data = response.read()
      etag = response.headers.get('ETag')
  except urllib.error.HTTPError as e:
    if e.code == 304 and saved is not None:
      return saved
    raise

  index = Index.from_json(json.loads(data))  # Check before saving.
  _write_atomic(saved_index_path(), data)
  if etag:
    _write_atomic(etag_path, etag.encode())
  elif etag_path.exists():
    etag_path.unlink()
  return index


def get_index(refresh: bool = True) -> Index:
  """Fetches the index if refresh, falling back to the saved one."""
  if not refresh:
    index = load_index()
    if index is not None:
      return index
  try:
    return fetch_index()
  except (OSError, ValueError) as e:  # URLError is an OSError.
    index = load_index()
    if index is None:
      raise
    logging.warning(f'Could not fetch the model index ({e}); using the saved one.')
    return index


def _model_dir(info: ModelInfo) -> pathlib.Path:
  # Keyed by hash so that a changed model never reuses a stale file.
  return cache_dir() / info.sha256[:16]


def downloaded_path(info: ModelInfo) -> tp.Optional[str]:
  path = _model_dir(info) / info.filename
  return str(path) if path.exists() else None


def downloaded_models() -> list[tuple[ModelInfo, str]]:
  """Downloaded models and their paths, whether or not still in the index."""
  models = []
  root = cache_dir()
  if not root.is_dir():
    return models
  for model_dir in sorted(root.iterdir()):
    info_path = model_dir / INFO_FILE
    if not info_path.is_file():
      continue
    try:
      info = ModelInfo.from_json(json.loads(info_path.read_text(encoding='utf-8')))
    except (ValueError, TypeError) as e:
      logging.warning(f'Ignoring {info_path}: {e}')
      continue
    path = downloaded_path(info)
    if path is not None:
      models.append((info, path))
  return models


def download(
    info: ModelInfo,
    progress: tp.Optional[ProgressFn] = None,
    cancel: tp.Optional[threading.Event] = None,
) -> str:
  """Downloads a model unless it's already downloaded; returns its path."""
  path = downloaded_path(info)
  if path is not None:
    return path

  dest = _model_dir(info) / info.filename
  dest.parent.mkdir(parents=True, exist_ok=True)
  partial = dest.with_name(dest.name + '.part')
  sha256 = hashlib.sha256()
  done = 0

  logging.info(f'Downloading {info.name} from {info.url}')
  request = urllib.request.Request(info.url, headers={'User-Agent': USER_AGENT})
  try:
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response, \
         open(partial, 'wb') as f:
      while chunk := response.read(1 << 20):
        if cancel is not None and cancel.is_set():
          raise DownloadCancelled(info.name)
        f.write(chunk)
        sha256.update(chunk)
        done += len(chunk)
        if progress is not None:
          progress(done, info.size)
  except BaseException:
    partial.unlink(missing_ok=True)
    raise

  if sha256.hexdigest() != info.sha256:
    partial.unlink()
    raise ValueError(
        f'{info.url} has sha256 {sha256.hexdigest()}, expected {info.sha256}.')

  # The entry goes first, so that a model file always has one.
  _write_atomic(dest.parent / INFO_FILE, json.dumps(info.to_json(), indent=2).encode())
  os.replace(partial, dest)
  return str(dest)


def delete_download(info: ModelInfo):
  model_dir = _model_dir(info)
  for path in (model_dir / info.filename, model_dir / INFO_FILE):
    path.unlink(missing_ok=True)
  try:
    model_dir.rmdir()
  except OSError:
    pass


def _find(spec: str, models: tp.Iterable[ModelInfo]) -> list[ModelInfo]:
  name, _, sha_prefix = spec.partition('@')
  return [m for m in models
          if m.name == name and m.sha256.startswith(sha_prefix.lower())]


def find_model(spec: str, index: tp.Optional[Index] = None) -> ModelInfo:
  """Looks up a model by name, or name@<sha256 prefix> for an exact file.

  Searches the index and downloaded models. Without an index, uses the saved
  one, fetching it if it doesn't have the model.
  """
  def candidates(index: tp.Optional[Index]) -> list[ModelInfo]:
    models = [] if index is None else list(index.models)
    models.extend(info for info, _ in downloaded_models())
    return _find(spec, models)

  matches = candidates(index or load_index())
  if index is None and not matches:
    try:
      index = fetch_index()
    except (OSError, ValueError) as e:
      logging.warning(f'Could not fetch the model index: {e}')
    else:
      matches = candidates(index)

  if not matches:
    raise ValueError(f'Unknown model "{spec}".')
  compatible = [m for m in matches if m.compatible()]
  if not compatible:
    raise ValueError(
        f'Model "{spec}" needs a different version of phillip (format '
        f'version {matches[0].format_version}, batch size {matches[0].batch_size}).')
  if len({m.sha256 for m in compatible}) > 1:
    # Prefer the index, which comes first, over older downloads.
    logging.info(f'Several models match "{spec}"; using {compatible[0].sha256[:16]}.')
  return compatible[0]


def _tqdm_progress(info: ModelInfo) -> ProgressFn:
  import tqdm
  bar = tqdm.tqdm(
      total=info.size, unit='B', unit_scale=True, unit_divisor=1024,
      desc=info.name)

  def progress(done: int, total: int):
    del total
    bar.update(done - bar.n)
    if done >= info.size:
      bar.close()

  return progress


def get_model_path(
    spec: str,
    progress: tp.Optional[ProgressFn] = None,
    index: tp.Optional[Index] = None,
) -> str:
  """Returns the local path to a published model, downloading it if needed.

  Shows a progress bar on the terminal unless `progress` is given.
  """
  info = find_model(spec, index)
  path = downloaded_path(info)
  if path is None:
    path = download(info, progress or _tqdm_progress(info))
  return path


def resolve_path(path: tp.Optional[str], model: tp.Optional[str]) -> str:
  """Resolves the --path/--model agent flags to a local path."""
  if path is not None and model is not None:
    raise ValueError('Pass either a model path or a model name, not both.')
  if model is not None:
    return get_model_path(model)
  if path is None:
    raise ValueError('Must pass a model path or a model name.')
  return path
