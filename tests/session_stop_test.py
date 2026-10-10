"""Tests stopping a session while Dolphin sends no frames.

Once the game is exited, Dolphin keeps the connection open but sends nothing,
so the session would wait forever for the next frame. A dummy ENet server
plays that Dolphin.
"""

import threading
import time
import unittest
from unittest import mock

import enet
import portpicker
from melee import slippstream

from slippi_ai import session


class SilentDolphin:
  """Accepts the connection, then sends no frames."""

  def __init__(self):
    port = portpicker.pick_unused_port()
    self._host = enet.Host(enet.Address(b'127.0.0.1', port), 4, 0, 0, 0)
    self._done = threading.Event()
    self._thread = threading.Thread(target=self._serve, daemon=True)
    self._thread.start()

    self.client = slippstream.SlippstreamClient('127.0.0.1', port)
    assert self.client.connect()

  def _serve(self):
    while not self._done.is_set():
      self._host.service(50)

  def interrupt(self):
    self.client.interrupt()

  def close(self):
    self.client.shutdown()
    self._done.set()
    self._thread.join()


class FakeSession:
  """A session whose Dolphin never sends a frame."""

  def __init__(self, config):
    del config
    self.dolphin = SilentDolphin()
    self.closed = False

  def frames(self, stop_event):
    del stop_event
    while True:
      self.dolphin.client.dispatch(polling_mode=False)
      yield

  def __enter__(self):
    return self

  def __exit__(self, *_):
    self.dolphin.close()
    self.closed = True


class SessionStopTest(unittest.TestCase):

  def test_stop_without_frames(self):
    sessions = []

    def make_session(config):
      sessions.append(FakeSession(config))
      return sessions[-1]

    stop_event = threading.Event()
    threading.Timer(1, stop_event.set).start()
    start = time.time()
    with mock.patch.object(session, 'Session', make_session):
      session.run_session(None, stop_event)
    elapsed = time.time() - start

    self.assertTrue(sessions[0].closed)
    self.assertLess(elapsed, 5)

  def test_disconnect_without_stop_raises(self):
    def make_session(config):
      fake = FakeSession(config)
      threading.Timer(0.5, fake.dolphin.interrupt).start()
      return fake

    with mock.patch.object(session, 'Session', make_session):
      with self.assertRaises(slippstream.EnetDisconnected):
        session.run_session(None, threading.Event())


if __name__ == '__main__':
  unittest.main()
