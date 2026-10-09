"""Tests finding and downloading app updates, from a local HTTP server."""

import functools
import hashlib
import http.server
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from slippi_ai import models
from slippi_ai.gui import updates


def release(tag: str, installer: bytes = b'', digest: bool = True, **kwargs) -> dict:
  version = tag.removeprefix(updates.TAG_PREFIX)
  asset = dict(
      name=f'phillip-setup-{version}.exe',
      size=len(installer),
      browser_download_url=f'http://example.com/{tag}.exe',
  )
  if digest:
    asset['digest'] = 'sha256:' + hashlib.sha256(installer).hexdigest()
  entry = dict(
      tag_name=tag, draft=False, prerelease=False,
      html_url=f'http://example.com/{tag}', assets=[asset])
  entry.update(kwargs)
  return entry


class LatestReleaseTest(unittest.TestCase):

  def test_newest_launcher_release(self):
    latest = updates.latest_release([
        release('launcher-v0.3.0'),
        release('v0.9.0'),  # A slippi-ai release.
        release('launcher-v0.10.0'),
        release('launcher-v0.4.0'),
        release('launcher-v1.0.0', draft=True),
        release('launcher-v1.1.0', prerelease=True),
        release('launcher-vbad'),
    ])
    self.assertEqual(latest.version, (0, 10, 0))
    self.assertEqual(latest.tag, 'launcher-v0.10.0')
    self.assertEqual(latest.installer_sha256, hashlib.sha256(b'').hexdigest())

  def test_no_digest(self):
    latest = updates.latest_release([release('launcher-v0.4.0', digest=False)])
    self.assertEqual(latest.version, (0, 4, 0))
    self.assertIsNone(latest.installer_url)

  def test_none(self):
    self.assertIsNone(updates.latest_release([release('v0.2.0')]))

  def test_current_version_parses(self):
    self.assertEqual(len(updates.current_version()), 3)

  def test_enabled(self):
    with mock.patch.dict(os.environ, {updates.CHECK_ENV_VAR: '1'}):
      self.assertTrue(updates.enabled())
    with mock.patch.dict(os.environ, {updates.CHECK_ENV_VAR: '0'}):
      self.assertFalse(updates.enabled())


class ServedTest(unittest.TestCase):

  def setUp(self):
    self.served_dir = tempfile.TemporaryDirectory()
    self.cache_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.served_dir.cleanup)
    self.addCleanup(self.cache_dir.cleanup)

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=self.served_dir.name)
    handler.log_message = lambda *_: None
    self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=self.server.serve_forever, daemon=True).start()
    self.addCleanup(self.server.server_close)
    self.addCleanup(self.server.shutdown)
    self.base_url = f'http://127.0.0.1:{self.server.server_port}/'

    env = mock.patch.dict(os.environ, {
        models.CACHE_ENV_VAR: os.path.join(self.cache_dir.name, 'models'),
        updates.RELEASES_URL_ENV_VAR: self.base_url + 'releases.json',
    })
    env.start()
    self.addCleanup(env.stop)

  def serve(self, filename: str, contents: bytes):
    with open(os.path.join(self.served_dir.name, filename), 'wb') as f:
      f.write(contents)

  def serve_releases(self, *versions: tuple[int, ...], installer: bytes = b'setup'):
    entries = []
    for v in versions:
      tag = updates.TAG_PREFIX + '.'.join(map(str, v))
      entry = release(tag, installer)
      entry['assets'][0]['browser_download_url'] = self.base_url + 'setup.exe'
      entries.append(entry)
    self.serve('releases.json', json.dumps(entries).encode())
    self.serve('setup.exe', installer)

  def test_check(self):
    current = updates.current_version()
    newer = current[:-1] + (current[-1] + 1,)
    self.serve_releases(current)
    self.assertIsNone(updates.check())
    self.serve_releases(current, newer)
    self.assertEqual(updates.check().version, newer)

  def test_download_installer(self):
    installer = os.urandom(1 << 20)
    self.serve_releases((99, 0, 0), installer=installer)
    release = updates.check()
    path = updates.download_installer(release)
    with open(path, 'rb') as f:
      self.assertEqual(f.read(), installer)
    self.assertTrue(path.endswith('phillip-setup-99.0.0.exe'))
    # Already downloaded.
    os.remove(os.path.join(self.served_dir.name, 'setup.exe'))
    self.assertEqual(updates.download_installer(release), path)

  def test_wrong_hash(self):
    self.serve_releases((99, 0, 0))
    release = updates.check()
    release.installer_sha256 = '0' * 64
    with self.assertRaisesRegex(ValueError, 'sha256'):
      updates.download_installer(release)
    self.assertEqual(os.listdir(updates.updates_dir()), [])


if __name__ == '__main__':
  unittest.main()
