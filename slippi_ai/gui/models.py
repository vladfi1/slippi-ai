"""Finds exported phillip models in a folder."""

import dataclasses
import os

import melee

from slippi_ai import eval_lib, onnx_policies


@dataclasses.dataclass
class Model:
  path: str
  name: str  # Path relative to the folder, without .onnx.
  summary: eval_lib.AgentSummary

  def plays(self, character: melee.Character) -> bool:
    return character in self.summary.characters

  def plays_against(self, character: melee.Character) -> bool:
    return character in self.summary.opponents


def scan(folder: str) -> tuple[list[Model], list[tuple[str, str]]]:
  """Returns the models under folder, and (path, error) for unreadable ones.

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
      models.append(Model(path=path, name=name.replace(os.sep, '/'), summary=summary))
  return models, errors
