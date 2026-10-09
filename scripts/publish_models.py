"""Publishes exported .onnx models and the model index to the Hugging Face Hub.

The models to publish, and their descriptions, are listed in
models/published.json:

  {
    "diamond": {
      "path": "diamond.onnx",     # relative to --models_dir
      "description": "..."
    }
  }

Files whose sha256 isn't in the index yet are uploaded in one commit, then
the index (slippi_ai/models.py) is rewritten in a second commit, with each
new entry's URL pinned to the first commit. Entries for models dropped from
the list are removed from the index, but their files stay on the Hub, so
existing downloads and pinned URLs keep working. Entries for other format
versions of a listed model are kept, for older installs.

  python scripts/publish_models.py --models_dir=onnx_models --dry_run

Requires the publish extra (pip install .[publish]) and `hf auth login` with
a write token.
"""

import datetime
import json
import os

from absl import app, flags

from slippi_ai import models

REPO = flags.DEFINE_string('repo', 'vladfi/phillip-models', 'Hugging Face model repo.')
MODELS_DIR = flags.DEFINE_string(
    'models_dir', 'onnx_models', 'Directory of exported .onnx models.')
PUBLISHED = flags.DEFINE_string(
    'published', 'models/published.json', 'Which models to publish.')
DRY_RUN = flags.DEFINE_boolean(
    'dry_run', False, 'Only print the new index and what would be uploaded.')

INDEX_FILE = f'index-v{models.INDEX_VERSION}.json'


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


def main(_):
  import huggingface_hub

  with open(PUBLISHED.value, encoding='utf-8') as f:
    published: dict[str, dict] = json.load(f)

  api = huggingface_hub.HfApi()
  repo = REPO.value
  current = load_current_index(api, repo)
  by_sha256 = {(m.name, m.sha256): m for m in current.models}
  today = datetime.date.today().isoformat()

  # Entries in published order; new ones get their URL after the upload.
  entries: list[models.ModelInfo] = []
  uploads: dict[str, str] = {}  # repo filename -> local path
  for name, spec in published.items():
    path = os.path.join(MODELS_DIR.value, spec['path'])
    filename = f'{name}.onnx'
    info = models.info_from_file(
        path, name, url=file_url(repo, '<commit>', filename),
        description=spec.get('description', ''), published=today)
    if not info.compatible():
      raise ValueError(f'{path} is not playable by this version of phillip.')

    old = by_sha256.get((name, info.sha256))
    if old is not None:
      info.url, info.published = old.url, old.published
    else:
      uploads[filename] = path
    entries.append(info)

    # Other format versions of this model, for older installs.
    entries.extend(
        m for m in current.models
        if m.name == name and m.format_version != info.format_version)

  dropped = sorted({m.name for m in current.models} - set(published))
  for filename, path in uploads.items():
    print(f'Upload {path} as {filename} ({os.path.getsize(path) / 2**20:.0f} MiB)')
  if dropped:
    print(f'Remove from the index: {", ".join(dropped)}')

  if DRY_RUN.value:
    print(json.dumps(models.Index(models=entries).to_json(), indent=2))
    return

  api.create_repo(repo, repo_type='model', exist_ok=True)

  if uploads:
    commit = api.create_commit(
        repo,
        operations=[
            huggingface_hub.CommitOperationAdd(filename, path)
            for filename, path in uploads.items()],
        commit_message=f'Upload {", ".join(uploads)}',
    )
    for info in entries:
      if info.url == file_url(repo, '<commit>', f'{info.name}.onnx'):
        info.url = file_url(repo, commit.oid, f'{info.name}.onnx')

  index = models.Index(
      models=entries,
      updated=datetime.datetime.now(datetime.timezone.utc).isoformat(
          timespec='seconds').replace('+00:00', 'Z'),
  )
  data = json.dumps(index.to_json(), indent=2) + '\n'
  api.upload_file(
      path_or_fileobj=data.encode(),
      path_in_repo=INDEX_FILE,
      repo_id=repo,
      commit_message=f'Update {INDEX_FILE}',
  )
  print(f'Published {len(entries)} models to {file_url(repo, "main", INDEX_FILE)}')


if __name__ == '__main__':
  app.run(main)
