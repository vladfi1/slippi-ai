"""Tests the published model index and downloads, from a local HTTP server."""

import functools
import hashlib
import http.server
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import melee

from slippi_ai import eval_lib, models, onnx_policies


class Handler(http.server.BaseHTTPRequestHandler):
  """Serves files from a directory, with ETags."""

  def __init__(self, *args, directory: str, requests: list, **kwargs):
    self.directory = directory
    self.requests = requests
    super().__init__(*args, **kwargs)

  def do_GET(self):
    self.requests.append((self.path, self.headers.get('If-None-Match')))
    path = os.path.join(self.directory, self.path.lstrip('/'))
    if not os.path.isfile(path):
      self.send_error(404)
      return
    with open(path, 'rb') as f:
      data = f.read()
    etag = '"%s"' % hashlib.sha256(data).hexdigest()[:16]
    if self.headers.get('If-None-Match') == etag:
      self.send_response(304)
      self.end_headers()
      return
    self.send_response(200)
    self.send_header('Content-Length', str(len(data)))
    self.send_header('ETag', etag)
    self.end_headers()
    self.wfile.write(data)

  def log_message(self, *args):
    pass


class ModelsTest(unittest.TestCase):

  def setUp(self):
    self.served_dir = tempfile.TemporaryDirectory()
    self.cache_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.served_dir.cleanup)
    self.addCleanup(self.cache_dir.cleanup)

    self.requests = []
    handler = functools.partial(
        Handler, directory=self.served_dir.name, requests=self.requests)
    self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    self.addCleanup(self.server.server_close)
    self.addCleanup(self.server.shutdown)
    self.base_url = f'http://127.0.0.1:{self.server.server_port}/'

    env = mock.patch.dict(os.environ, {
        models.CACHE_ENV_VAR: self.cache_dir.name,
        models.INDEX_URL_ENV_VAR: self.base_url + 'index-v1.json',
    })
    env.start()
    self.addCleanup(env.stop)

    self.contents = self.serve('fox.onnx', os.urandom(3 << 20))

  def serve(self, filename: str, contents: bytes) -> bytes:
    with open(os.path.join(self.served_dir.name, filename), 'wb') as f:
      f.write(contents)
    return contents

  def unserve(self, filename: str):
    os.remove(os.path.join(self.served_dir.name, filename))

  def entry(self, **overrides) -> dict:
    entry = dict(
        name='fox',
        url=self.base_url + 'fox.onnx',
        sha256=hashlib.sha256(self.contents).hexdigest(),
        size=len(self.contents),
        format_version=onnx_policies.FORMAT_VERSION,
        batch_size=models.PLAY_BATCH_SIZE,
        agent_type='RL',
        characters=['FOX'],
        opponents=['FALCO', 'FOX'],
        delay=21,
    )
    entry.update(overrides)
    return entry

  def serve_index(self, *entries: dict):
    index = dict(updated='2026-10-09T00:00:00Z', models=list(entries))
    self.serve('index-v1.json', json.dumps(index).encode())

  def test_fetch_and_save_index(self):
    self.assertIsNone(models.load_index())
    self.serve_index(self.entry(future_field='ignored'))

    index = models.fetch_index()
    self.assertEqual([m.name for m in index.models], ['fox'])
    self.assertEqual(models.load_index(), index)

    # Unchanged: the saved index is reused (304).
    self.assertEqual(models.fetch_index(), index)
    self.assertIsNotNone(self.requests[-1][1])

    # Offline: falls back to the saved index.
    self.unserve('index-v1.json')
    with self.assertRaises(OSError):
      models.fetch_index()
    self.assertEqual(models.get_index(), index)

  def test_index_updates(self):
    self.serve_index(self.entry())
    models.fetch_index()
    self.serve_index(self.entry(), self.entry(name='fox2'))
    self.assertEqual(
        [m.name for m in models.fetch_index().models], ['fox', 'fox2'])
    self.assertEqual(len(models.load_index().models), 2)

  def test_skips_incomplete_entries(self):
    entry = self.entry()
    del entry['sha256']
    self.serve_index(entry, self.entry(name='fox2'))
    self.assertEqual([m.name for m in models.fetch_index().models], ['fox2'])

  def test_no_index_url(self):
    with mock.patch.dict(os.environ), \
         mock.patch.object(models, 'DEFAULT_INDEX_URL', None):
      del os.environ[models.INDEX_URL_ENV_VAR]
      with self.assertRaisesRegex(ValueError, 'No model index'):
        models.get_index()

  def test_summary(self):
    info = models.ModelInfo.from_json(self.entry())
    self.assertEqual(info.summary(), eval_lib.AgentSummary(
        type=eval_lib.AgentType.RL,
        delay=21,
        characters=[melee.Character.FOX],
        opponents=[melee.Character.FALCO, melee.Character.FOX],
    ))

  def test_compatible(self):
    self.assertTrue(models.ModelInfo.from_json(self.entry()).compatible())
    self.assertFalse(models.ModelInfo.from_json(
        self.entry(format_version=onnx_policies.FORMAT_VERSION + 1)).compatible())
    self.assertFalse(models.ModelInfo.from_json(
        self.entry(batch_size=None)).compatible())

  def test_download_and_cache(self):
    self.serve_index(self.entry())

    progress = []
    path = models.get_model_path(
        'fox', progress=lambda done, total: progress.append(done))
    with open(path, 'rb') as f:
      self.assertEqual(f.read(), self.contents)
    self.assertTrue(path.startswith(self.cache_dir.name))
    self.assertEqual(progress[-1], len(self.contents))

    # Downloaded: no download even if the server is gone, or the model was
    # removed from the index.
    self.unserve('fox.onnx')
    self.serve_index()
    models.fetch_index()
    self.assertEqual(models.get_model_path('fox'), path)
    [(info, downloaded)] = models.downloaded_models()
    self.assertEqual((info.name, downloaded), ('fox', path))

    models.delete_download(info)
    self.assertEqual(models.downloaded_models(), [])
    self.assertIsNone(models.downloaded_path(info))

  def test_wrong_hash(self):
    info = models.ModelInfo.from_json(self.entry(sha256='0' * 64))
    with self.assertRaisesRegex(ValueError, 'sha256'):
      models.download(info)
    self.assertEqual(models.downloaded_models(), [])
    self.assertEqual(os.listdir(os.path.join(self.cache_dir.name, '0' * 16)), [])

  def test_cancel(self):
    info = models.ModelInfo.from_json(self.entry())
    cancel = threading.Event()

    def progress(done, total):
      del done, total
      cancel.set()

    with self.assertRaises(models.DownloadCancelled):
      models.download(info, progress, cancel)
    self.assertIsNone(models.downloaded_path(info))
    self.assertEqual(os.listdir(os.path.join(
        self.cache_dir.name, info.sha256[:16])), [])

  def test_find_model(self):
    other = self.serve('fox-old.onnx', os.urandom(1 << 10))
    old_sha256 = hashlib.sha256(other).hexdigest()
    self.serve_index(
        self.entry(),
        self.entry(url=self.base_url + 'fox-old.onnx', sha256=old_sha256,
                   size=len(other)),
        self.entry(name='future', format_version=onnx_policies.FORMAT_VERSION + 1),
    )

    # Not fetched yet: find_model fetches the index.
    self.assertEqual(models.find_model('fox').sha256, self.entry()['sha256'])
    self.assertEqual(
        models.find_model(f'fox@{old_sha256[:8].upper()}').sha256, old_sha256)
    with self.assertRaisesRegex(ValueError, 'Unknown model'):
      models.find_model('falco')
    with self.assertRaisesRegex(ValueError, 'Unknown model'):
      models.find_model('fox@ffffffff')
    with self.assertRaisesRegex(ValueError, 'different version'):
      models.find_model('future')

  def test_find_model_refetches(self):
    self.serve_index()
    models.fetch_index()
    self.serve_index(self.entry())
    # Not in the saved index, so it's fetched again.
    self.assertEqual(models.find_model('fox').name, 'fox')

  def test_resolve_path(self):
    self.assertEqual(models.resolve_path('a.onnx', None), 'a.onnx')
    with self.assertRaises(ValueError):
      models.resolve_path('a.onnx', 'fox')
    with self.assertRaises(ValueError):
      models.resolve_path(None, None)


if __name__ == '__main__':
  unittest.main()
