"""A window for playing against phillip locally in Slippi Dolphin."""

import logging
import os
import subprocess
import sys
import threading
import typing as tp

import melee
from PySide6 import QtCore, QtGui, QtWidgets

from slippi_ai import session, utils
from slippi_ai.gui import models as models_lib, runner, settings as settings_lib

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
      error = None
    except Exception as e:
      result, error = None, e
    try:
      signals.done.emit(result, error)
    except RuntimeError:
      pass  # The app quit while fn was running.

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


def _set_items(
    combo: QtWidgets.QComboBox,
    items: list[tuple[str, tp.Any]],
    preferred: tp.Any,
):
  """Replaces a combo box's items, selecting `preferred` if it's there."""
  combo.blockSignals(True)
  combo.clear()
  for text, data in items:
    combo.addItem(text, data)
  combo.setCurrentIndex(max(0, combo.findData(preferred)))
  combo.blockSignals(False)


def _same_path(a: str, b: str) -> bool:
  return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


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
    self._models: list[models_lib.Model] = []
    self._models_scan_id = 0  # Ignores results from earlier scans.
    self._models_scanning = False
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
    self.resize(720, 780)

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

    self.models_row = PathRow(choose_dir=True)
    self.models_row.changed.connect(self._scan_models)
    self.models_status = QtWidgets.QLabel()
    self.models_status.setWordWrap(True)
    form.addRow('Models folder', self.models_row)
    form.addRow('', self.models_status)

    # The settings remember what the user picked, even while no model in the
    # folder matches it.
    self.character_combo = QtWidgets.QComboBox()
    self.character_combo.activated.connect(self._character_chosen)
    self.opponent_filter_combo = QtWidgets.QComboBox()
    self.opponent_filter_combo.activated.connect(self._opponent_filter_chosen)
    form.addRow('phillip\'s character', self.character_combo)
    form.addRow('Opponent\'s character', self.opponent_filter_combo)

    self.model_list = QtWidgets.QTreeWidget()
    self.model_list.setHeaderLabels(['Model', 'Reaction delay', 'Trained against'])
    self.model_list.setRootIsDecorated(False)
    self.model_list.setMinimumHeight(110)
    self.model_list.itemSelectionChanged.connect(self._model_chosen)
    form.addRow('Model', self.model_list)
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
    self.models_row.set_path(s.models_dir)

  def _save_settings(self):
    s = self.settings
    s.dolphin_path = self.dolphin_row.path()
    s.iso_path = self.iso_row.path()
    s.models_dir = self.models_row.path()
    model = self._selected_model()
    if model is not None:
      s.model_path = model.path
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

  def _scan_models(self, folder: str):
    self._models_scan_id += 1
    scan_id = self._models_scan_id
    self._models = []
    self._models_scanning = False
    self.models_status.setToolTip('')
    self._update_character_filter()

    if not folder:
      _status(self.models_status,
              'Choose the folder with your phillip models (.onnx files).', False)
      return
    if not os.path.isdir(folder):
      _status(self.models_status, 'Folder not found.', False)
      return

    self._models_scanning = True
    _status(self.models_status, 'Looking for models...', None)
    self._update_controls()

    def on_done(result, error):
      if scan_id != self._models_scan_id:
        return  # Another folder was chosen meanwhile.
      self._models_scanning = False
      if error is not None:
        _status(self.models_status, f'Could not read the folder: {error}', False)
      else:
        self._models, errors = result
        count = len(self._models)
        if count:
          text = f'Found {count} model{"" if count == 1 else "s"}.'
        else:
          text = 'No phillip models (.onnx files) found.'
        if errors:
          text += f' Skipped {len(errors)} .onnx file(s); hover for details.'
          self.models_status.setToolTip(
              '\n'.join(f'{path}: {message}' for path, message in errors))
        _status(self.models_status, text, bool(count))
      self._update_character_filter()

    run_in_thread(lambda: models_lib.scan(folder), on_done)

  def _character(self) -> tp.Optional[melee.Character]:
    name = self.character_combo.currentData()
    return melee.Character[name] if name else None

  def _opponent_filter(self) -> tp.Optional[melee.Character]:
    name = self.opponent_filter_combo.currentData()
    return melee.Character[name] if name else None

  def _selected_model(self) -> tp.Optional[models_lib.Model]:
    items = self.model_list.selectedItems()
    if not items:
      return None
    return items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)

  def _character_chosen(self):
    self.settings.character = self.character_combo.currentData()
    self._update_opponent_filter()

  def _opponent_filter_chosen(self):
    self.settings.opponent_character = self.opponent_filter_combo.currentData()
    self._update_model_list()

  def _model_chosen(self):
    model = self._selected_model()
    if model is not None:
      self.settings.model_path = model.path
    self._update_controls()

  def _update_character_filter(self):
    characters = {c for m in self._models for c in m.summary.characters}
    _set_items(
        self.character_combo,
        [(character_name(c), c.name)
         for c in sorted(characters, key=character_name)],
        self.settings.character)
    self._update_opponent_filter()

  def _update_opponent_filter(self):
    """Lists the opponents of the models that play phillip's character."""
    character = self._character()
    opponents = {
        c for m in self._models if character and m.plays(character)
        for c in m.summary.opponents}
    _set_items(
        self.opponent_filter_combo,
        [('Any', '')] + [(character_name(c), c.name)
                         for c in sorted(opponents, key=character_name)],
        self.settings.opponent_character)
    self._update_model_list()

  def _update_model_list(self):
    character = self._character()
    opponent = self._opponent_filter()
    matching = [
        m for m in self._models
        if character and m.plays(character)
        and (opponent is None or m.plays_against(opponent))]

    self.model_list.blockSignals(True)
    self.model_list.clear()
    selected = None
    for model in matching:
      summary = model.summary
      opponents = sorted(summary.opponents, key=character_name)
      if len(opponents) > 3:
        against = f'{len(opponents)} characters'
      else:
        against = ', '.join(character_name(c) for c in opponents)
      frames = 'frame' if summary.delay == 1 else 'frames'
      item = QtWidgets.QTreeWidgetItem(
          [model.name, f'{summary.delay} {frames}', against])
      item.setData(0, QtCore.Qt.ItemDataRole.UserRole, model)
      item.setToolTip(0, model.path)
      item.setToolTip(2, ', '.join(character_name(c) for c in opponents))
      self.model_list.addTopLevelItem(item)
      if selected is None and _same_path(model.path, self.settings.model_path):
        selected = item
    if selected is None and matching:
      selected = self.model_list.topLevelItem(0)
    if selected is not None:
      self.model_list.setCurrentItem(selected)
    for column in range(self.model_list.columnCount()):
      self.model_list.resizeColumnToContents(column)
    self.model_list.blockSignals(False)
    self._update_controls()

  # Running

  def _running(self) -> bool:
    return self._process is not None

  def _update_controls(self):
    running = self._running()
    for widget in (self.dolphin_row, self.iso_row, self.models_row,
                   self.character_combo, self.opponent_filter_combo,
                   self.model_list, self.human_radio, self.cpu_radio,
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
        self._dolphin_ok and self._iso_ok and not self._models_scanning and
        self._character() is not None and self._selected_model() is not None)
    if running:
      self.start_button.setText('Stop')
      self.start_button.setEnabled(self._stop_requested_at is None)
    else:
      self.start_button.setText('Start')
      self.start_button.setEnabled(ready)

  def _session_config(self) -> session.SessionConfig:
    model = self._selected_model()
    assert model is not None
    summary = model.summary

    defaults = utils.map_nt(lambda item: item.default, session.player_flags())
    human = self.human_radio.isChecked()
    other_port = self.port_combo.currentData() if human else 1
    ai_port, = (p for p in session.PORTS if p != other_port)

    ai = utils.map_nt(lambda x: x, defaults)
    ai['type'] = 'ai'
    ai['character'] = self._character()
    ai['ai']['path'] = model.path

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
