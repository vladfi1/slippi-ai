"""Runs a session in a child process, so a hung or crashing Dolphin can't
take down the GUI and a stuck session can still be killed."""

import logging
import logging.handlers
import multiprocessing
import queue
import sys
import typing as tp

from slippi_ai import session

# Seconds to wait after asking the session to stop before killing it. The
# session checks for stop once per frame, but not while starting Dolphin.
STOP_TIMEOUT = 10

_LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'


def _child_main(
    config: session.SessionConfig,
    stop_event,
    log_queue: multiprocessing.Queue,
):
  root = logging.getLogger()
  root.handlers[:] = [logging.handlers.QueueHandler(log_queue)]
  root.setLevel(logging.INFO)

  try:
    session.run_session(config, stop_event)
  except Exception:
    logging.exception('The session failed.')
    sys.exit(1)


class SessionProcess:
  """A session in a child process. Not thread-safe; poll from one thread."""

  def __init__(self, config: session.SessionConfig):
    # Spawn on every platform; forking a process with Qt threads is unsafe.
    context = multiprocessing.get_context('spawn')
    self._stop_event = context.Event()
    self._log_queue = context.Queue()
    # Not a daemon: libmelee's slippstream starts its own worker process,
    # which daemons can't. The GUI stops the session when it closes.
    self._process = context.Process(
        target=_child_main,
        args=(config, self._stop_event, self._log_queue),
    )
    self._process.start()
    self._formatter = logging.Formatter(_LOG_FORMAT, datefmt='%H:%M:%S')

  def poll_logs(self) -> list[tuple[int, str]]:
    """Returns (level, message) for the records logged since the last poll."""
    records = []
    while True:
      try:
        record = self._log_queue.get_nowait()
      except queue.Empty:
        return records
      records.append((record.levelno, self._formatter.format(record)))

  def is_alive(self) -> bool:
    return self._process.is_alive()

  @property
  def exitcode(self) -> tp.Optional[int]:
    return self._process.exitcode

  def request_stop(self):
    self._stop_event.set()

  def kill(self):
    """Kills the session. Dolphin may be left running."""
    self._process.kill()
    self._process.join()
