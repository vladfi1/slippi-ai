"""Manages the published models on the Hugging Face Hub.

The index on the Hub (slippi_ai/models.py) is the list of published models,
so publishing needs no commit here; the Hub keeps the history.

  # Publish a model, or a new version of one (named after the file by default).
  python scripts/publish_models.py add deployed_models/diamond.onnx \
    --description="Plays and faces all characters."
  python scripts/publish_models.py add a.onnx b.onnx

  # Make the index match a folder: publish its new and changed models, and
  # remove published models that aren't in it.
  python scripts/publish_models.py sync onnx_models

  python scripts/publish_models.py remove diamond
  python scripts/publish_models.py describe diamond --description="..."
  python scripts/publish_models.py list

A new file is uploaded as <name>.onnx, then the index is rewritten in a second
commit, with the file's URL pinned to the first. Re-adding a model keeps its
description unless --description is given. Entries for other format versions
of the model are kept, for older installs. Removing a model only removes it
from the index; its files stay on the Hub, so existing downloads and pinned
URLs keep working.

Requires the publish extra (pip install .[publish]) and `hf auth login` with
a write token.
"""

import datetime
import json
import os

from absl import app, flags

from slippi_ai import models

REPO = flags.DEFINE_string('repo', 'vladfi/phillip-models', 'Hugging Face model repo.')
NAME = flags.DEFINE_string(
    'name', None, 'Name for the model being added; defaults to the file name.')
DESCRIPTION = flags.DEFINE_string(
    'description', None, 'Description for add or describe.')
DRY_RUN = flags.DEFINE_boolean(
    'dry_run', False, 'Only print the new index and what would be uploaded.')

INDEX_FILE = f'index-v{models.INDEX_VERSION}.json'
PENDING = '<commit>'


def file_url(repo: str, revision: str, filename: str) -> str:
  return f'https://huggingface.co/{repo}/resolve/{revision}/{filename}'


def load_current_index(api, repo: str) -> models.Index:
  import huggingface_hub
  try:
    path = api.hf_hub_download(repo, INDEX_FILE, revision='main')
  except (huggingface_hub.errors.RepositoryNotFoundError,
          huggingface_hub.errors.EntryNotFoundError,
          huggingface_hub.errors.RevisionNotFoundError):
    return models.Index(models=[])
  with open(path, encoding='utf-8') as f:
    return models.Index.from_json(json.load(f))


def model_name(path: str) -> str:
  return os.path.splitext(os.path.basename(path))[0]


def add(
    index: models.Index,
    named_paths: list[tuple[str, str]],
    repo: str,
    skip_incompatible: bool = False,
) -> dict[str, str]:
  """Adds models to the index; returns the files to upload (repo name -> path)."""
  uploads = {}
  for name, path in named_paths:
    filename = f'{name}.onnx'
    info = models.info_from_file(
        path, name, url=file_url(repo, PENDING, filename),
        published=datetime.date.today().isoformat())
    if not info.compatible():
      message = f'{path} is not playable by this version of phillip.'
      if skip_incompatible:
        print(f'Skipping {message}')
        continue
      raise ValueError(message)

    old = [m for m in index.models if m.name == name]
    same_format = [m for m in old if m.format_version == info.format_version]
    info.description = DESCRIPTION.value or next(
        (m.description for m in same_format or old), '')
    if same_format and same_format[0].sha256 == info.sha256:
      print(f'{name} is already published.')
      info.url, info.published = same_format[0].url, same_format[0].published
    else:
      uploads[filename] = path

    # Replace the entry for this format version, in place for a new version.
    position = index.models.index(same_format[0]) if same_format else len(index.models)
    index.models = [m for m in index.models if m not in same_format]
    index.models.insert(min(position, len(index.models)), info)
  return uploads


def find(index: models.Index, name: str) -> list[models.ModelInfo]:
  matches = [m for m in index.models if m.name == name]
  if not matches:
    raise app.UsageError(f'No published model "{name}".')
  return matches


def print_index(index: models.Index):
  for m in index.models:
    characters = ', '.join(m.characters) if len(m.characters) <= 3 else (
        f'{len(m.characters)} characters')
    print(f'{m.name}: {characters}, delay {m.delay}, '
          f'{m.size / 2**20:.0f} MiB, format {m.format_version}, '
          f'published {m.published}')
    if m.description:
      print(f'  {m.description}')


def main(argv):
  if len(argv) < 2:
    raise app.UsageError('Usage: publish_models.py add|sync|remove|describe|list ...')
  command, args = argv[1], argv[2:]

  import huggingface_hub
  api = huggingface_hub.HfApi()
  repo = REPO.value
  index = load_current_index(api, repo)

  before = index.to_json()
  uploads = {}
  if command == 'list':
    print_index(index)
    return
  elif command == 'add':
    if not args:
      raise app.UsageError('add needs .onnx files.')
    if NAME.value is not None and len(args) != 1:
      raise app.UsageError('--name needs exactly one model.')
    uploads = add(index, [(NAME.value or model_name(path), path) for path in args], repo)
    message = f'Add {", ".join(f.removesuffix(".onnx") for f in uploads)}'
  elif command == 'sync':
    if len(args) != 1 or NAME.value is not None or DESCRIPTION.value is not None:
      raise app.UsageError('Usage: sync <folder>')
    folder = args[0]
    named_paths = [
        (model_name(f), os.path.join(folder, f))
        for f in sorted(os.listdir(folder)) if f.endswith('.onnx')]
    uploads = add(index, named_paths, repo, skip_incompatible=True)
    removed = sorted({m.name for m in index.models} - {n for n, _ in named_paths})
    index.models = [m for m in index.models if m.name not in removed]
    for name in removed:
      print(f'Remove {name}')
    message = f'Sync: add {", ".join(f.removesuffix(".onnx") for f in uploads) or "none"}, '               f'remove {", ".join(removed) or "none"}'
  elif command == 'remove':
    if not args:
      raise app.UsageError('remove needs model names.')
    for name in args:
      removed = find(index, name)
      index.models = [m for m in index.models if m not in removed]
    message = f'Remove {", ".join(args)}'
  elif command == 'describe':
    if len(args) != 1 or DESCRIPTION.value is None:
      raise app.UsageError('Usage: describe <name> --description=...')
    for m in find(index, args[0]):
      m.description = DESCRIPTION.value
    message = f'Describe {args[0]}'
  else:
    raise app.UsageError(f'Unknown command "{command}".')

  for filename, path in uploads.items():
    print(f'Upload {path} as {filename} ({os.path.getsize(path) / 2**20:.0f} MiB)')

  if index.to_json() == before:
    print('Nothing to change.')
    return

  if DRY_RUN.value:
    print(json.dumps(index.to_json(), indent=2))
    return

  if uploads:
    api.create_repo(repo, repo_type='model', exist_ok=True)
    commit = api.create_commit(
        repo,
        operations=[
            huggingface_hub.CommitOperationAdd(filename, path)
            for filename, path in uploads.items()],
        commit_message=f'Upload {", ".join(uploads)}',
    )
    for info in index.models:
      info.url = info.url.replace(f'/resolve/{PENDING}/', f'/resolve/{commit.oid}/')

  index.updated = datetime.datetime.now(datetime.timezone.utc).isoformat(
      timespec='seconds').replace('+00:00', 'Z')
  data = json.dumps(index.to_json(), indent=2) + '\n'
  api.upload_file(
      path_or_fileobj=data.encode(),
      path_in_repo=INDEX_FILE,
      repo_id=repo,
      commit_message=f'{message} ({INDEX_FILE})',
  )
  print_index(index)


if __name__ == '__main__':
  app.run(main)
