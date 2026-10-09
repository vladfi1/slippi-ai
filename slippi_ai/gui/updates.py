"""Updates for the installed phillip app, from GitHub releases.

App releases are tagged launcher-v<version> (slippi_ai/gui/version.py), and
carry the installer, phillip-setup-<version>.exe. GitHub publishes each
asset's sha256, which the download is checked against. The installer is run
silently with /relaunch=1: it waits for the app to exit (APP_MUTEX),
upgrades it in place and starts it again (packaging/slippi_ai.iss).
"""

import dataclasses
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import typing as tp
import urllib.request

from slippi_ai import models
from slippi_ai.gui import version

RELEASES_URL = 'https://api.github.com/repos/vladfi1/slippi-ai/releases?per_page=50'
RELEASES_URL_ENV_VAR = 'PHILLIP_RELEASES_URL'
# 1 checks for updates even when not installed, e.g. to test; 0 never checks.
CHECK_ENV_VAR = 'PHILLIP_CHECK_UPDATES'

TAG_PREFIX = 'launcher-v'
INSTALLER_PATTERN = re.compile(r'phillip-setup-.*\.exe')
# Held while the app runs; the installer waits for it to be released. Must
# match AppMutex in packaging/slippi_ai.iss.
APP_MUTEX = 'phillip-launcher'

Version = tuple[int, ...]


@dataclasses.dataclass
class Release:
  version: Version
  tag: str
  page_url: str
  # None if the release has no installer with a published sha256.
  installer_url: tp.Optional[str] = None
  installer_size: int = 0
  installer_sha256: tp.Optional[str] = None

  @property
  def version_str(self) -> str:
    return '.'.join(map(str, self.version))


def parse_version(text: str) -> tp.Optional[Version]:
  match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', text)
  return None if match is None else tuple(int(x) for x in match.groups())


def current_version() -> Version:
  parsed = parse_version(version.VERSION)
  assert parsed is not None, version.VERSION
  return parsed


def enabled() -> bool:
  """Whether to check for updates: only in the installed (frozen) app."""
  setting = os.environ.get(CHECK_ENV_VAR)
  if setting is not None:
    return setting == '1'
  return sys.platform == 'win32' and getattr(sys, 'frozen', False)


def fetch_releases(url: tp.Optional[str] = None) -> list[dict]:
  url = url or os.environ.get(RELEASES_URL_ENV_VAR, RELEASES_URL)
  request = urllib.request.Request(url, headers={
      'User-Agent': models.USER_AGENT,
      'Accept': 'application/vnd.github+json',
  })
  with urllib.request.urlopen(request, timeout=models.TIMEOUT) as response:
    return json.load(response)


def latest_release(releases: list[dict]) -> tp.Optional[Release]:
  """The newest published app release, if any."""
  latest = None
  for entry in releases:
    tag = entry.get('tag_name', '')
    if entry.get('draft') or entry.get('prerelease') or not tag.startswith(TAG_PREFIX):
      continue
    parsed = parse_version(tag.removeprefix(TAG_PREFIX))
    if parsed is None or (latest is not None and parsed <= latest.version):
      continue
    release = Release(version=parsed, tag=tag, page_url=entry.get('html_url', ''))
    for asset in entry.get('assets', []):
      digest = asset.get('digest') or ''
      if INSTALLER_PATTERN.fullmatch(asset.get('name', '')) and digest.startswith('sha256:'):
        release.installer_url = asset['browser_download_url']
        release.installer_size = asset.get('size', 0)
        release.installer_sha256 = digest.removeprefix('sha256:')
        break
    latest = release
  return latest


def check() -> tp.Optional[Release]:
  """Returns a release newer than this app, or None."""
  release = latest_release(fetch_releases())
  if release is not None and release.version > current_version():
    return release
  return None


def updates_dir() -> pathlib.Path:
  return models.cache_dir().parent / 'updates'


def download_installer(
    release: Release,
    progress: tp.Optional[models.ProgressFn] = None,
    cancel: tp.Optional[threading.Event] = None,
) -> str:
  assert release.installer_url and release.installer_sha256
  dest = updates_dir() / f'phillip-setup-{release.version_str}.exe'
  if dest.exists() and models.sha256_file(str(dest)) == release.installer_sha256:
    return str(dest)
  partial = models.download_file(
      release.installer_url, dest, release.installer_sha256,
      release.installer_size, progress, cancel)
  os.replace(partial, dest)
  return str(dest)


def run_installer(path: str):
  """Starts the installer; the app must exit for it to proceed."""
  flags = 0
  if sys.platform == 'win32':
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
  subprocess.Popen(
      [path, '/SILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/relaunch=1'],
      creationflags=flags, close_fds=True)


_mutex = None


def hold_app_mutex():
  """Marks the app as running, for the installer (Windows only)."""
  global _mutex
  if sys.platform != 'win32' or _mutex is not None:
    return
  import ctypes
  _mutex = ctypes.windll.kernel32.CreateMutexW(None, False, APP_MUTEX)
