"""A window for playing against phillip locally in Slippi Dolphin."""

import logging
import os
import subprocess
import sys
import threading
import typing as tp

import melee
from PySide6 import QtCore, QtGui, QtWidgets

from slippi_ai import models as model_index, onnx_policies, session, utils
from slippi_ai.gui import (
    models as models_lib, runner, settings as settings_lib, updates, version)

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


def _source_text(model: models_lib.Model) -> str:
  if model.source is models_lib.Source.ONLINE:
    return f'Download ({model.info.size / 1e6:.0f} MB)'
  if model.source is models_lib.Source.DOWNLOADED:
    return 'Downloaded'
  return 'Your folder'


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
    self.setWindowTitle(f'Play phillip {version.VERSION}')
    self.settings = settings_lib.load()

    self._dolphin_ok = False
    self._iso_ok = False
    self._iso_checked_path = None
    # Published models, from the index and downloads, then local ones.
    self._models: list[models_lib.Model] = []
    self._index: tp.Optional[model_index.Index] = None
    self._index_fetching = False
    self._downloaded: list[tuple[model_index.ModelInfo, str]] = []
    self._download_cancel: tp.Optional[threading.Event] = None
    self._local_models: list[models_lib.Model] = []
    self._models_scan_id = 0  # Ignores results from earlier scans.
    self._models_scanning = False
    self._process: tp.Optional[runner.SessionProcess] = None
    self._update: tp.Optional[updates.Release] = None
    self._update_cancel: tp.Optional[threading.Event] = None
    self._stop_requested_at: tp.Optional[QtCore.QElapsedTimer] = None

    central = QtWidgets.QWidget()
    self.setCentralWidget(central)
    layout = QtWidgets.QVBoxLayout(central)
    layout.addWidget(self._update_bar())
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
    self._load_published()
    self._fetch_index()
    self._find_providers()
    self._check_for_updates()
    self._update_controls()
    self.resize(720, 800)

  # Layout

  def _update_bar(self) -> QtWidgets.QWidget:
    """Offers a newer version of the app; hidden until there is one."""
    self.update_bar = QtWidgets.QFrame()
    self.update_bar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
    self.update_label = QtWidgets.QLabel()
    self.update_label.setOpenExternalLinks(True)
    self.update_label.setWordWrap(True)
    self.update_button = QtWidgets.QPushButton('Update')
    self.update_button.clicked.connect(self._update_clicked)
    later = QtWidgets.QPushButton('Later')
    later.clicked.connect(self.update_bar.hide)
    self._update_later = later

    row = QtWidgets.QHBoxLayout(self.update_bar)
    row.addWidget(self.update_label, 1)
    row.addWidget(self.update_button)
    row.addWidget(later)
    self.update_bar.hide()
    return self.update_bar

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

    self.copy_settings_check = QtWidgets.QCheckBox(
        'Use my Slippi Dolphin settings (graphics, audio, controllers)')
    self.copy_settings_check.setToolTip(
        "Starts Dolphin with a copy of Slippi Dolphin's settings; changes "
        "made during the session aren't saved. If unchecked, Dolphin uses "
        'default settings and your port uses a GameCube controller adapter.')
    form.addRow('', self.copy_settings_check)
    return group

  def _phillip_group(self) -> QtWidgets.QGroupBox:
    group = QtWidgets.QGroupBox('phillip')
    form = QtWidgets.QFormLayout(group)

    self.index_status = QtWidgets.QLabel()
    self.index_status.setWordWrap(True)
    self.show_online_check = QtWidgets.QCheckBox(
        'Show models that need to be downloaded')
    self.show_online_check.toggled.connect(self._show_online_toggled)
    form.addRow('Published models', self.index_status)
    form.addRow('', self.show_online_check)

    self.models_row = PathRow(choose_dir=True)
    self.models_row.changed.connect(self._scan_models)
    self.models_status = QtWidgets.QLabel()
    self.models_status.setWordWrap(True)
    form.addRow('Your models folder', self.models_row)
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
    self.model_list.setHeaderLabels(
        ['Model', 'Reaction delay', 'Trained against', 'Status'])
    self.model_list.setRootIsDecorated(False)
    self.model_list.setMinimumHeight(110)
    self.model_list.itemSelectionChanged.connect(self._model_chosen)
    self.model_list.setContextMenuPolicy(
        QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    self.model_list.customContextMenuRequested.connect(self._model_menu)
    form.addRow('Model', self.model_list)

    # Filled in by _find_providers.
    self.provider_combo = QtWidgets.QComboBox()
    self.provider_combo.addItem('Looking for devices...', '')
    self.provider_combo.activated.connect(self._provider_chosen)
    form.addRow('Run on', self.provider_combo)
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

    self.cpu_character_combo = QtWidgets.QComboBox()
    for c in CPU_CHARACTERS:
      self.cpu_character_combo.addItem(character_name(c), c.name)
    self.cpu_level_spin = QtWidgets.QSpinBox()
    self.cpu_level_spin.setRange(1, 9)
    self.cpu_level_spin.setPrefix('Level ')

    grid.addWidget(self.human_radio, 0, 0)
    grid.addWidget(self.port_combo, 0, 1)
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
    self.copy_settings_check.setChecked(s.copy_dolphin_settings)
    self.cpu_character_combo.setCurrentIndex(
        max(0, self.cpu_character_combo.findData(s.cpu_character)))
    self.cpu_level_spin.setValue(s.cpu_level)
    self.show_online_check.blockSignals(True)
    self.show_online_check.setChecked(s.show_online_models)
    self.show_online_check.blockSignals(False)

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
      s.model = model.key
      s.character = self.character_combo.currentData()
    s.show_online_models = self.show_online_check.isChecked()
    s.opponent = 'human' if self.human_radio.isChecked() else 'cpu'
    s.human_port = self.port_combo.currentData()
    s.copy_dolphin_settings = self.copy_settings_check.isChecked()
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
    self._local_models = []
    self._models_scanning = False
    self.models_status.setToolTip('')
    self._refresh_models()

    if not folder:
      _status(self.models_status,
              'Optional: a folder with your own phillip models (.onnx files).',
              None)
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
        self._local_models, errors = result
        count = len(self._local_models)
        if count:
          text = f'Found {count} model{"" if count == 1 else "s"}.'
        else:
          text = 'No phillip models (.onnx files) found.'
        if errors:
          text += f' Skipped {len(errors)} .onnx file(s); hover for details.'
          self.models_status.setToolTip(
              '\n'.join(f'{path}: {message}' for path, message in errors))
        _status(self.models_status, text, bool(count))
      self._refresh_models()

    run_in_thread(lambda: models_lib.scan(folder), on_done)

  # App updates

  def _check_for_updates(self):
    if not updates.enabled():
      return

    def on_done(release, error):
      if error is not None:
        logging.warning('Could not check for updates: %s', error)
      elif release is not None:
        self._update = release
        self._show_update_available()
        if release.installer_url is None:
          self.update_button.setText('Download')
        self.update_bar.show()
        self._update_controls()

    run_in_thread(updates.check, on_done)

  def _show_update_available(self):
    release = self._update
    self.update_label.setText(
        f'phillip {release.version_str} is available. '
        f'<a href="{release.page_url}">What\'s new</a>')

  def _update_clicked(self):
    release = self._update
    assert release is not None
    if self._update_cancel is not None:
      self._update_cancel.set()
      return
    if release.installer_url is None:
      # No installer with a published hash: download it from the page.
      QtGui.QDesktopServices.openUrl(QtCore.QUrl(release.page_url))
      return

    cancel = threading.Event()
    self._update_cancel = cancel
    self.update_button.setText('Cancel')
    self._update_later.setEnabled(False)
    size = f'{release.installer_size / 1e6:.0f} MB'
    self.update_label.setText(f'Downloading phillip {release.version_str} ({size})...')
    self._update_controls()

    def on_progress(fraction):
      if not cancel.is_set():
        self.update_label.setText(
            f'Downloading phillip {release.version_str}... {fraction:.0%} of {size}')

    def on_done(path, error):
      self._update_cancel = None
      self.update_button.setText('Update')
      self._update_later.setEnabled(True)
      if error is None:
        try:
          updates.run_installer(path)
        except OSError as e:
          error = e
        else:
          self.update_label.setText(
              f'Installing phillip {release.version_str}; it will restart.')
          self.close()
          return
      if isinstance(error, model_index.DownloadCancelled):
        self._show_update_available()
      else:
        logging.warning('Could not update: %s', error)
        self.update_label.setText(
            f'Could not update to phillip {release.version_str}: {error}')
      self._update_controls()

    run_in_thread(
        lambda progress: updates.download_installer(
            release, lambda done, total: progress(done / max(total, 1)), cancel),
        on_done, on_progress)

  # Published models

  def _load_published(self):
    """Shows the saved index and downloads; fast, so on the GUI thread."""
    self._index = model_index.load_index()
    self._downloaded = model_index.downloaded_models()
    self._refresh_models()

  def _fetch_index(self):
    self._index_fetching = True
    self._update_index_status()

    def on_done(index, error):
      self._index_fetching = False
      if error is not None:
        logging.warning('Could not fetch the model index: %s', error)
      else:
        self._index = index
      self._update_index_status(error)
      self._refresh_models()

    run_in_thread(model_index.fetch_index, on_done)

  def _update_index_status(self, error: tp.Optional[BaseException] = None):
    published = models_lib.published(self._index, self._downloaded)
    downloaded = sum(m.source is models_lib.Source.DOWNLOADED for m in published)
    count = f'{len(published)} model{"" if len(published) == 1 else "s"}'
    count += f', {downloaded} downloaded.'
    self.index_status.setToolTip('' if error is None else str(error))
    if self._index_fetching:
      _status(self.index_status, 'Checking for new models...', None)
    elif error is not None:
      if self._index is None:
        _status(self.index_status,
                "Couldn't get the list of published models; check your "
                'internet connection. Hover for details.', False)
      else:
        _status(self.index_status,
                "Couldn't reach the list of published models; showing the "
                f'list from {self._index.updated[:10]}. {count}', False)
    else:
      text = count
      incompatible = models_lib.incompatible_count(self._index)
      if incompatible:
        text += f' {incompatible} more need a newer version of phillip.'
      _status(self.index_status, text, True)

  def _refresh_models(self):
    published = models_lib.published(self._index, self._downloaded)
    if not self.show_online_check.isChecked():
      published = [
          m for m in published if m.source is not models_lib.Source.ONLINE]
    self._models = published + self._local_models
    self._update_character_filter()

  def _show_online_toggled(self):
    self.settings.show_online_models = self.show_online_check.isChecked()
    self._refresh_models()

  def _downloads_changed(self):
    self._downloaded = model_index.downloaded_models()
    if not self._index_fetching:
      self._update_index_status()
    self._refresh_models()

  def _model_menu(self, pos: QtCore.QPoint):
    item = self.model_list.itemAt(pos)
    if item is None or self._busy():
      return
    model: models_lib.Model = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
    if model.source is not models_lib.Source.DOWNLOADED:
      return
    menu = QtWidgets.QMenu(self)
    delete = menu.addAction('Delete download')
    if menu.exec(self.model_list.viewport().mapToGlobal(pos)) is delete:
      try:
        model_index.delete_download(model.info)
      except OSError as e:
        QtWidgets.QMessageBox.warning(
            self, 'Delete download', f'Could not delete {model.name}: {e}')
      self._downloads_changed()

  def _download_and_start(self, model: models_lib.Model):
    info = model.info
    assert info is not None
    cancel = threading.Event()
    self._download_cancel = cancel
    size = f'{info.size / 1e6:.0f} MB'
    self.status.setText(f'Downloading {info.name} ({size})...')
    self._update_controls()

    def download(progress):
      path = model_index.download(
          info, lambda done, total: progress(done / max(total, 1)), cancel)
      models_lib.delete_other_versions(info)
      return path

    def on_progress(fraction):
      if self._download_cancel is cancel and not cancel.is_set():
        self.status.setText(f'Downloading {info.name}... {fraction:.0%} of {size}')

    def on_done(path, error):
      self._download_cancel = None
      self._downloads_changed()
      if isinstance(error, model_index.DownloadCancelled):
        self.status.setText('Download cancelled.')
      elif error is not None:
        logging.warning('Could not download %s: %s', info.name, error)
        self.status.setText(f'Could not download {info.name}: {error}')
      else:
        self.status.setText('')
        selected = self._selected_model()
        if selected is not None and selected.path == path:
          self._start_session()
      self._update_controls()

    run_in_thread(download, on_done, on_progress)

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
      self.settings.model = model.key
    self._update_controls()

  def _find_providers(self):
    # Can take a while: Windows ML may download a provider the first time.
    def find():
      return onnx_policies.provider_choices(), onnx_policies.default_providers()

    def on_done(result, error):
      if error is not None:
        logging.warning('Could not list devices: %s', error)
        choices, default = [], [onnx_policies.CPU]
      else:
        choices, default = result
      automatic = onnx_policies.PROVIDER_NAMES.get(default[0], default[0])
      _set_items(
          self.provider_combo,
          [(f'Automatic: {automatic}', '')]
          + [(choice.label, choice.provider) for choice in choices],
          self.settings.provider)

    run_in_thread(find, on_done)

  def _provider_chosen(self):
    self.settings.provider = self.provider_combo.currentData()

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
          [model.name, f'{summary.delay} {frames}', against, _source_text(model)])
      item.setData(0, QtCore.Qt.ItemDataRole.UserRole, model)
      tooltip = [model.info.description] if model.info and model.info.description else []
      tooltip.append(model.path or model.info.url)
      item.setToolTip(0, '\n'.join(tooltip))
      item.setToolTip(2, ', '.join(character_name(c) for c in opponents))
      self.model_list.addTopLevelItem(item)
      if selected is None and self._is_chosen(model):
        selected = item
    if selected is None and matching:
      selected = self.model_list.topLevelItem(0)
    if selected is not None:
      self.model_list.setCurrentItem(selected)
    for column in range(self.model_list.columnCount()):
      self.model_list.resizeColumnToContents(column)
    self.model_list.blockSignals(False)
    self._update_controls()

  def _is_chosen(self, model: models_lib.Model) -> bool:
    chosen = self.settings.model
    if model.source is models_lib.Source.LOCAL:
      return bool(chosen) and _same_path(model.path, chosen)
    return model.key == chosen

  # Running

  def _running(self) -> bool:
    return self._process is not None

  def _downloading(self) -> bool:
    return self._download_cancel is not None

  def _updating(self) -> bool:
    return self._update_cancel is not None

  def _busy(self) -> bool:
    return self._running() or self._downloading() or self._updating()

  def _update_controls(self):
    running = self._running()
    busy = self._busy()
    for widget in (self.dolphin_row, self.iso_row, self.models_row,
                   self.show_online_check,
                   self.character_combo, self.opponent_filter_combo,
                   self.model_list, self.provider_combo,
                   self.human_radio, self.cpu_radio,
                   self.port_combo, self.copy_settings_check,
                   self.cpu_character_combo, self.cpu_level_spin):
      widget.setEnabled(not busy)

    if not busy:
      human = self.human_radio.isChecked()
      self.port_combo.setEnabled(human)
      self.cpu_character_combo.setEnabled(not human)
      self.cpu_level_spin.setEnabled(not human)

    ready = (
        self._dolphin_ok and self._iso_ok and not self._models_scanning and
        self._character() is not None and self._selected_model() is not None)
    # Updating closes the app, so not during a session or a model download.
    self.update_button.setEnabled(
        self._updating() or not (running or self._downloading()))

    if running:
      self.start_button.setText('Stop')
      self.start_button.setEnabled(self._stop_requested_at is None)
    elif self._updating():
      self.start_button.setText('Start')
      self.start_button.setEnabled(False)
    elif self._downloading():
      self.start_button.setText('Cancel download')
      self.start_button.setEnabled(not self._download_cancel.is_set())
    else:
      model = self._selected_model()
      online = model is not None and model.path is None
      self.start_button.setText('Download and start' if online else 'Start')
      self.start_button.setEnabled(ready)

  def _session_config(self) -> session.SessionConfig:
    model = self._selected_model()
    assert model is not None and model.path is not None
    summary = model.summary

    defaults = utils.map_nt(lambda item: item.default, session.player_flags())
    human = self.human_radio.isChecked()
    other_port = self.port_combo.currentData() if human else 1
    ai_port, = (p for p in session.PORTS if p != other_port)

    ai = utils.map_nt(lambda x: x, defaults)
    ai['type'] = 'ai'
    ai['character'] = self._character()
    ai['ai']['path'] = model.path
    provider = self.provider_combo.currentData()
    if provider:
      # The CPU runs whatever the chosen provider can't.
      ai['ai']['onnx']['providers'] = list(dict.fromkeys(
          [provider, onnx_policies.CPU]))

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
    dolphin.copy_home_directory = self.copy_settings_check.isChecked()
    if dolphin.copy_home_directory:
      # Keep the user's display settings, e.g. fullscreen.
      dolphin.fullscreen = None

    return session.SessionConfig(
        players={ai_port: ai, other_port: other}, dolphin=dolphin)

  def _start_or_stop(self):
    if self._running():
      self._stop()
    elif self._downloading():
      self._download_cancel.set()
      self.status.setText('Cancelling download...')
      self._update_controls()
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

    model = self._selected_model()
    if model is not None and model.path is None:
      self._download_and_start(model)
    else:
      self._start_session()

  def _start_session(self):
    self._save_settings()
    config = self._session_config()
    self.log.clear()
    # For bug reports: which build the log is from.
    self._append_log(logging.INFO, version.build_info())
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
    for cancel in (self._download_cancel, self._update_cancel):
      if cancel is not None:
        cancel.set()
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
  app.setApplicationName('phillip')
  updates.hold_app_mutex()
  window = MainWindow()
  window.show()
  return app.exec()
