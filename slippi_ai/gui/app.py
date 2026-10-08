"""A window for playing against phillip locally in Slippi Dolphin."""

import logging
import os
import subprocess
import sys
import threading
import typing as tp

import melee
from PySide6 import QtCore, QtGui, QtWidgets

from slippi_ai import eval_lib, saving, session, utils
from slippi_ai.gui import runner, settings as settings_lib

_CHARACTER_NAMES = {
    melee.Character.CPTFALCON: 'Captain Falcon',
    melee.Character.DK: 'Donkey Kong',
    melee.Character.DOC: 'Dr. Mario',
    melee.Character.GAMEANDWATCH: 'Mr. Game & Watch',
    melee.Character.POPO: 'Ice Climbers',
    melee.Character.YLINK: 'Young Link',
}


def character_name(character: melee.Character) -> str:
  return _CHARACTER_NAMES.get(character, character.name.title())


# Characters that can be picked for the in-game CPU.
CPU_CHARACTERS = sorted(
    (c for c in melee.Character
     if c.value < melee.Character.WIREFRAME_MALE.value
     and c is not melee.Character.NANA),
    key=character_name)


# Signals of running threads, kept alive until the threads finish.
_live_signals = set()


class _Signals(QtCore.QObject):
  done = QtCore.Signal(object, object)  # result, exception
  progress = QtCore.Signal(float)


def run_in_thread(
    fn: tp.Callable[..., tp.Any],
    on_done: tp.Callable[[tp.Any, tp.Optional[BaseException]], None],
    on_progress: tp.Optional[tp.Callable[[float], None]] = None,
) -> None:
  """Runs fn in a thread and calls on_done(result, error) on the GUI thread.

  If on_progress is given, fn is called with a progress callback.
  """
  signals = _Signals()
  signals.done.connect(on_done)
  if on_progress is not None:
    signals.progress.connect(on_progress)

  def target():
    try:
      if on_progress is None:
        result = fn()
      else:
        result = fn(signals.progress.emit)
    except Exception as e:
      signals.done.emit(None, e)
    else:
      signals.done.emit(result, None)

  _live_signals.add(signals)
  signals.done.connect(lambda *_: _live_signals.discard(signals))
  threading.Thread(target=target, daemon=True).start()


def running_dolphins() -> list[str]:
  """Names of running Dolphin processes (Windows only)."""
  if sys.platform != 'win32':
    return []
  try:
    output = subprocess.run(
        ['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW, timeout=10).stdout
  except (OSError, subprocess.SubprocessError):
    return []
  names = set()
  for line in output.splitlines():
    name = line.split('","')[0].strip('"')
    if 'dolphin' in name.lower():
      names.add(name)
  return sorted(names)


def _status(label: QtWidgets.QLabel, text: str, ok: tp.Optional[bool]):
  color = {True: 'green', False: '#c00000', None: 'gray'}[ok]
  label.setText(text)
  label.setStyleSheet(f'color: {color}')


class PathRow(QtWidgets.QWidget):
  """A path field with a Browse button."""

  changed = QtCore.Signal(str)

  def __init__(self, choose_dir: bool, file_filter: str = ''):
    super().__init__()
    self._choose_dir = choose_dir
    self._filter = file_filter

    self.edit = QtWidgets.QLineEdit()
    self.edit.editingFinished.connect(lambda: self.changed.emit(self.path()))
    browse = QtWidgets.QPushButton('Browse...')
    browse.clicked.connect(self._browse)

    layout = QtWidgets.QHBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(self.edit)
    layout.addWidget(browse)

  def path(self) -> str:
    return self.edit.text().strip().strip('"')

  def set_path(self, path: str):
    self.edit.setText(path)
    self.changed.emit(self.path())

  def _browse(self):
    start = self.path() or os.path.expanduser('~')
    if self._choose_dir:
      path = QtWidgets.QFileDialog.getExistingDirectory(self, dir=start)
    else:
      path, _ = QtWidgets.QFileDialog.getOpenFileName(
          self, dir=os.path.dirname(start), filter=self._filter)
    if path:
      self.set_path(os.path.normpath(path))


class MainWindow(QtWidgets.QMainWindow):

  def __init__(self):
    super().__init__()
    self.setWindowTitle('Play phillip')
    self.settings = settings_lib.load()

    self._dolphin_ok = False
    self._iso_ok = False
    self._iso_checked_path = None
    self._model_path = None
    self._model_summary: tp.Optional[eval_lib.AgentSummary] = None
    self._model_loading = False
    self._process: tp.Optional[runner.SessionProcess] = None
    self._stop_requested_at: tp.Optional[QtCore.QElapsedTimer] = None

    central = QtWidgets.QWidget()
    self.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    layout.addWidget(self._melee_group())
    layout.addWidget(self._phillip_group())
    layout.addWidget(self._opponent_group())

    buttons = QtWidgets.QHBoxLayout()
    self.start_button = QtWidgets.QPushButton('Start')
    self.start_button.clicked.connect(self._start_or_stop)
    self.start_button.setMinimumHeight(32)
    self.status = QtWidgets.QLabel()
    buttons.addWidget(self.start_button)
    buttons.addWidget(self.status, 1)
    layout.addLayout(buttons)

    self.log = QtWidgets.QPlainTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumBlockCount(5000)
    self.log.setFont(QtGui.QFontDatabase.systemFont(
        QtGui.QFontDatabase.SystemFont.FixedFont))
    layout.addWidget(self.log, 1)

    self._poll_timer = QtCore.QTimer(self)
    self._poll_timer.setInterval(100)
    self._poll_timer.timeout.connect(self._poll)

    self._load_settings()
    self._update_controls()
    self.resize(720, 640)

  # Layout

  def _melee_group(self) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox('Melee')
    form = QtWidgets.QFormLayout(group)

    self.dolphin_row = PathRow(choose_dir=True)
    self.dolphin_row.changed.connect(self._check_dolphin)
    self.dolphin_status = QtWidgets.QLabel()
    form.addRow('Slippi Dolphin folder', self.dolphin_row)
    form.addRow('', self.dolphin_status)

    self.iso_row = PathRow(
        choose_dir=False, file_filter='Melee ISO (*.iso);;All files (*)')
    self.iso_row.changed.connect(self._check_iso)
    self.iso_status = QtWidgets.QLabel()
    form.addRow('Melee 1.02 ISO', self.iso_row)
    form.addRow('', self.iso_status)
    return group

  def _phillip_group(self) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox('phillip')
    form = QtWidgets.QFormLayout(group)

    self.model_row = PathRow(
        choose_dir=False, file_filter='phillip models (*.onnx);;All files (*)')
    self.model_row.changed.connect(self._load_model)
    self.model_status = QtWidgets.QLabel()
    self.model_status.setWordWrap(True)
    form.addRow('Model file', self.model_row)
    form.addRow('', self.model_status)

    self.character_combo = QtWidgets.QComboBox()
    form.addRow('Character', self.character_combo)
    return group

  def _opponent_group(self) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox('Opponent')
    grid = QtWidgets.QGridLayout(group)

    self.human_radio = QtWidgets.QRadioButton('Me, in port')
    self.cpu_radio = QtWidgets.QRadioButton('In-game CPU (watch)')
    self.human_radio.toggled.connect(self._update_controls)

    self.port_combo = QtWidgets.QComboBox()
    for port in session.PORTS:
      self.port_combo.addItem(str(port), port)

    self.controller_check = QtWidgets.QCheckBox(
        'Use my Slippi Dolphin controller settings')
    self.controller_check.setToolTip(
        'Copies the controller setup from Slippi Dolphin. If unchecked, your '
        'port uses a GameCube controller adapter.')

    self.cpu_character_combo = QtWidgets.QComboBox()
    for c in CPU_CHARACTERS:
      self.cpu_character_combo.addItem(character_name(c), c.name)
    self.cpu_level_spin = QtWidgets.QSpinBox()
    self.cpu_level_spin.setRange(1, 9)
    self.cpu_level_spin.setPrefix('Level ')

    grid.addWidget(self.human_radio, 0, 0)
    grid.addWidget(self.port_combo, 0, 1)
    grid.addWidget(self.controller_check, 0, 2, 1, 2)
    grid.addWidget(self.cpu_radio, 1, 0)
    grid.addWidget(self.cpu_character_combo, 1, 1, 1, 2)
    grid.addWidget(self.cpu_level_spin, 1, 3)

    note = QtWidgets.QLabel(
        'Close other Dolphin windows before starting: they can block '
        'phillip\'s inputs. A GameCube controller adapter needs the same '
        'driver setup as for Slippi.')
    note.setWordWrap(True)
    note.setStyleSheet('color: gray')
    grid.addWidget(note, 2, 0, 1, 4)
    grid.setColumnStretch(4, 1)
    return group

  # Settings

  def _load_settings(self):
    s = self.settings
    if not s.dolphin_path or not s.iso_path:
      dolphin_path, iso_path = settings_lib.detect_slippi()
      s.dolphin_path = s.dolphin_path or dolphin_path
      s.iso_path = s.iso_path or iso_path

    (self.human_radio if s.opponent == 'human' else self.cpu_radio).setChecked(True)
    self.port_combo.setCurrentIndex(max(0, self.port_combo.findData(s.human_port)))
    self.controller_check.setChecked(s.use_slippi_controller_settings)
    self.cpu_character_combo.setCurrentIndex(
        max(0, self.cpu_character_combo.findData(s.cpu_character)))
    self.cpu_level_spin.setValue(s.cpu_level)

    self.dolphin_row.set_path(s.dolphin_path)
    self.iso_row.set_path(s.iso_path)
    self.model_row.set_path(s.model_path)

  def _save_settings(self):
    s = self.settings
    s.dolphin_path = self.dolphin_row.path()
    s.iso_path = self.iso_row.path()
    s.model_path = self.model_row.path()
    if self.character_combo.currentData():
      s.character = self.character_combo.currentData()
    s.opponent = 'human' if self.human_radio.isChecked() else 'cpu'
    s.human_port = self.port_combo.currentData()
    s.use_slippi_controller_settings = self.controller_check.isChecked()
    s.cpu_character = self.cpu_character_combo.currentData()
    s.cpu_level = self.cpu_level_spin.value()
    try:
      settings_lib.save(s)
    except OSError as e:
      logging.warning(f'Could not save settings: {e}')

  # Checks

  def _check_dolphin(self, path: str):
    error = settings_lib.check_dolphin(path)
    self._dolphin_ok = error is None
    _status(self.dolphin_status, error or 'Found Slippi Dolphin.', self._dolphin_ok)
    self._update_controls()

  def _check_iso(self, path: str):
    if path == self._iso_checked_path:
      return
    self._iso_checked_path = path
    self._iso_ok = False

    if not path:
      _status(self.iso_status, 'Choose your Melee ISO.', False)
    elif not os.path.isfile(path):
      _status(self.iso_status, 'File not found.', False)
    else:
      # The ISO can still be used if the hash differs, so allow starting.
      self._iso_ok = True
      _status(self.iso_status, 'Checking ISO...', None)

      def on_done(md5, error):
        if path != self._iso_checked_path:
          return
        if error is not None:
          self._iso_ok = False
          _status(self.iso_status, f'Could not read ISO: {error}', False)
        elif md5 == settings_lib.MELEE_102_MD5:
          _status(self.iso_status, 'Melee 1.02 (NTSC).', True)
        else:
          _status(
              self.iso_status,
              'Not a vanilla NTSC 1.02 ISO; phillip may not work with it.',
              False)
        self._update_controls()

      def on_progress(fraction):
        if path == self._iso_checked_path and self._iso_ok:
          _status(self.iso_status, f'Checking ISO... {fraction:.0%}', None)

      run_in_thread(
          lambda progress: settings_lib.iso_md5(path, progress),
          on_done, on_progress)
    self._update_controls()

  def _load_model(self, path: str):
    if path == self._model_path:
      return
    self._model_path = path
    self._model_summary = None
    self.character_combo.clear()

    if not path:
      _status(self.model_status, 'Choose an exported phillip model (.onnx).', False)
      self._update_controls()
      return
    if not os.path.isfile(path):
      _status(self.model_status, 'File not found.', False)
      self._update_controls()
      return

    self._model_loading = True
    _status(self.model_status, 'Loading model...', None)
    self._update_controls()

    def load():
      state = saving.load_state_from_disk(path)
      return eval_lib.AgentSummary.from_state(state)

    def on_done(summary: eval_lib.AgentSummary, error):
      if path != self._model_path:
        return  # Another model was chosen meanwhile.
      self._model_loading = False
      if error is not None:
        _status(self.model_status, f'Could not load model: {error}', False)
      else:
        self._model_summary = summary
        self.character_combo.clear()
        for c in summary.characters:
          self.character_combo.addItem(character_name(c), c.name)
        index = self.character_combo.findData(self.settings.character)
        self.character_combo.setCurrentIndex(max(0, index))
        frames = 'frame' if summary.delay == 1 else 'frames'
        _status(
            self.model_status,
            f'Reaction delay: {summary.delay} {frames}.', True)
      self._update_controls()

    run_in_thread(load, on_done)

  # Running

  def _running(self) -> bool:
    return self._process is not None

  def _update_controls(self):
    running = self._running()
    for widget in (self.dolphin_row, self.iso_row, self.model_row,
                   self.character_combo, self.human_radio, self.cpu_radio,
                   self.port_combo, self.controller_check,
                   self.cpu_character_combo, self.cpu_level_spin):
      widget.setEnabled(not running)

    if not running:
      human = self.human_radio.isChecked()
      self.port_combo.setEnabled(human)
      self.controller_check.setEnabled(human)
      self.cpu_character_combo.setEnabled(not human)
      self.cpu_level_spin.setEnabled(not human)

    ready = (
        self._dolphin_ok and self._iso_ok and
        self._model_summary is not None and not self._model_loading)
    if running:
      self.start_button.setText('Stop')
      self.start_button.setEnabled(self._stop_requested_at is None)
    else:
      self.start_button.setText('Start')
      self.start_button.setEnabled(ready)

  def _session_config(self) -> session.SessionConfig:
    summary = self._model_summary
    assert summary is not None

    defaults = utils.map_nt(lambda item: item.default, session.player_flags())
    human = self.human_radio.isChecked()
    other_port = self.port_combo.currentData() if human else 1
    ai_port, = (p for p in session.PORTS if p != other_port)

    ai = utils.map_nt(lambda x: x, defaults)
    ai['type'] = 'ai'
    ai['character'] = melee.Character[self.character_combo.currentData()]
    ai['ai']['path'] = self.model_row.path()

    other = utils.map_nt(lambda x: x, defaults)
    if human:
      other['type'] = 'human'
    else:
      other['type'] = 'cpu'
      other['character'] = melee.Character[self.cpu_character_combo.currentData()]
      other['level'] = self.cpu_level_spin.value()

    dolphin = session.default_dolphin_config()
    dolphin.path = self.dolphin_row.path()
    dolphin.iso = self.iso_row.path()
    # Models with less delay than the default online delay need less.
    dolphin.online_delay = min(dolphin.online_delay, summary.delay)
    dolphin.copy_home_directory = human and self.controller_check.isChecked()

    return session.SessionConfig(
        players={ai_port: ai, other_port: other}, dolphin=dolphin)

  def _start_or_stop(self):
    if self._running():
      self._stop()
    else:
      self._start()

  def _start(self):
    dolphins = running_dolphins()
    if dolphins:
      answer = QtWidgets.QMessageBox.warning(
          self, 'Dolphin is running',
          f'{", ".join(dolphins)} is already running, which can block '
          'phillip\'s inputs. Close it first.\n\nStart anyway?',
          QtWidgets.QMessageBox.StandardButton.Yes
          | QtWidgets.QMessageBox.StandardButton.No)
      if answer != QtWidgets.QMessageBox.StandardButton.Yes:
        return

    self._save_settings()
    config = self._session_config()
    self.log.clear()
    self._append_log(logging.INFO, 'Starting Dolphin...')
    self._process = runner.SessionProcess(config)
    self._stop_requested_at = None
    self.status.setText('Running. Close Dolphin or press Stop to end.')
    self._poll_timer.start()
    self._update_controls()

  def _stop(self):
    if self._process is None:
      return
    self._process.request_stop()
    self._stop_requested_at = QtCore.QElapsedTimer()
    self._stop_requested_at.start()
    self.status.setText('Stopping...')
    self._update_controls()

  def _poll(self):
    process = self._process
    if process is None:
      return

    for level, message in process.poll_logs():
      self._append_log(level, message)

    if process.is_alive():
      if (self._stop_requested_at is not None and
          self._stop_requested_at.elapsed() > runner.STOP_TIMEOUT * 1000):
        self._append_log(logging.WARNING, 'Session did not stop; killing it.')
        process.kill()
      else:
        return

    for level, message in process.poll_logs():
      self._append_log(level, message)

    stopped = self._stop_requested_at is not None
    if process.exitcode == 0 or stopped:
      self.status.setText('Stopped.')
    else:
      self.status.setText(
          f'The session ended with an error (exit code {process.exitcode}); '
          'see the log.')
    self._process = None
    self._stop_requested_at = None
    self._poll_timer.stop()
    self._update_controls()

  def _append_log(self, level: int, message: str):
    if level >= logging.ERROR:
      color = '#c00000'
    elif level >= logging.WARNING:
      color = '#b06000'
    else:
      color = None
    cursor = self.log.textCursor()
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
    fmt = QtGui.QTextCharFormat()
    if color:
      fmt.setForeground(QtGui.QColor(color))
    cursor.insertText(message + '\n', fmt)
    self.log.setTextCursor(cursor)
    self.log.ensureCursorVisible()

  def closeEvent(self, event: QtGui.QCloseEvent):
    self._save_settings()
    if self._process is not None:
      self._process.request_stop()
      deadline = QtCore.QDeadlineTimer(runner.STOP_TIMEOUT * 1000)
      while self._process.is_alive() and not deadline.hasExpired():
        QtCore.QThread.msleep(100)
      if self._process.is_alive():
        self._process.kill()
    super().closeEvent(event)


def main():
  logging.basicConfig(level=logging.INFO)
  app = QtWidgets.QApplication(sys.argv)
  app.setApplicationName('slippi-ai')
  window = MainWindow()
  window.show()
  return app.exec()
