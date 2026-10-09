"""The models the GUI lists: published ones, downloaded or not, and local files."""

import dataclasses
import enum
import os
import typing as tp

import melee

from slippi_ai import eval_lib, models as models_lib, onnx_policies


class Source(enum.Enum):
  ONLINE = enum.auto()  # Published, not downloaded yet.
  DOWNLOADED = enum.auto()  # Published, downloaded.
  LOCAL = enum.auto()  # A file in the user's models folder.


@dataclasses.dataclass
class Model:
  name: str  # For local files, the path relative to the folder, without .onnx.
  summary: eval_lib.AgentSummary
  source: Source
  path: tp.Optional[str] = None  # None until downloaded.
  info: tp.Optional[models_lib.ModelInfo] = None  # For published models.

  @property
  def key(self) -> str:
    """Identifies the model in the settings."""
    if self.info is not None:
      # By name, so that the choice carries over to updated versions.
      return f'published:{self.name}'
    assert self.path is not None
    return local_key(self.path)

  def plays(self, character: melee.Character) -> bool:
    return character in self.summary.characters

  def plays_against(self, character: melee.Character) -> bool:
    return character in self.summary.opponents


def local_key(path: str) -> str:
  return os.path.normcase(os.path.normpath(path))


# Play runs one game, and models for it are exported with a fixed batch size
# of 1 (needed for CUDA graphs). Others, e.g. for evals, are skipped.
PLAY_BATCH_SIZE = models_lib.PLAY_BATCH_SIZE


def scan(folder: str) -> tuple[list[Model], list[tuple[str, str]]]:
  """Returns the models under folder, and (path, reason) for skipped ones.

  Only reads the models' metadata, so it's fast even for large models.
  """
  models = []
  errors = []
  for root, dirs, files in os.walk(folder):
    dirs.sort()
    for file in sorted(files):
      if not file.endswith('.onnx'):
        continue
      path = os.path.join(root, file)
      name = os.path.relpath(path, folder).removesuffix('.onnx')
      try:
        state = onnx_policies.load_state_from_disk(path, load_model=False)
        summary = eval_lib.AgentSummary.from_state(state)
      except Exception as e:
        errors.append((path, str(e)))
        continue
      batch_size = state['onnx_metadata']['batch_size']
      if batch_size != PLAY_BATCH_SIZE:
        size = 'a dynamic batch size' if batch_size is None else f'batch size {batch_size}'
        errors.append((path, f'Exported with {size}, not for play.'))
        continue
      models.append(Model(
          name=name.replace(os.sep, '/'), summary=summary,
          source=Source.LOCAL, path=path))
  return models, errors


def published(
    index: tp.Optional[models_lib.Index],
    downloaded: list[tuple[models_lib.ModelInfo, str]],
) -> list[Model]:
  """Published models, in index order, then downloads no longer in the index.

  A model in the index shows once: as downloaded if this version of it is,
  otherwise as online, hiding downloads of its older versions.
  """
  paths = {info.sha256: path for info, path in downloaded}
  models = []
  names = set()

  def add(info: models_lib.ModelInfo):
    if not info.compatible() or info.name in names:
      return
    try:
      summary = info.summary()
    except KeyError:  # A character this version of melee doesn't know.
      return
    path = paths.get(info.sha256)
    models.append(Model(
        name=info.name, summary=summary,
        source=Source.ONLINE if path is None else Source.DOWNLOADED,
        path=path, info=info))
    names.add(info.name)

  for info in [] if index is None else index.models:
    add(info)
  for info, _ in downloaded:
    add(info)
  return models


def incompatible_count(index: tp.Optional[models_lib.Index]) -> int:
  """How many published models need another version of phillip."""
  if index is None:
    return 0
  compatible = {m.name for m in index.models if m.compatible()}
  return len({m.name for m in index.models} - compatible)


def delete_other_versions(info: models_lib.ModelInfo):
  """Deletes downloads of other versions of a just-downloaded model."""
  for other, _ in models_lib.downloaded_models():
    if other.name == info.name and other.sha256 != info.sha256:
      models_lib.delete_download(other)
