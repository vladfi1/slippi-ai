"""Windows ML: execution providers for the PC's hardware, for onnxruntime.

With the winml extra (onnxruntime-windowsml), Windows downloads and keeps
up to date execution providers for the PC's hardware, such as TensorRT-RTX
for NVIDIA RTX GPUs on Windows 11 24H2+. They must be registered with
onnxruntime before use, and are then chosen by device rather than by name;
see onnx_policies.SessionRunner. Without them onnxruntime-windowsml still
runs models on the CPU (or DirectML), on any Windows version.

Starting Windows ML needs the Windows App SDK Runtime matching the wasdk-
packages (see setup.cfg).
"""

import atexit
import ctypes
import importlib.util
import logging
import os
import sys
import threading

TENSORRT_RTX = 'NvTensorRTRTXExecutionProvider'

_lock = threading.Lock()
_registered: list[str] | None = None
# Shuts down the Windows App SDK, at exit.
_bootstrap = None


def installed() -> bool:
  """Whether the Windows ML packages are installed."""
  return sys.platform == 'win32' and importlib.util.find_spec('winui3') is not None


def initialize(download: bool = True) -> list[str]:
  """Registers Windows ML's execution providers with onnxruntime, once.

  Args:
    download: Download providers for this PC's hardware that aren't
      installed yet. This can take a while the first time.

  Returns:
    The names of the registered providers; empty if Windows ML isn't
    installed or can't start.
  """
  global _registered, _bootstrap
  with _lock:
    if _registered is not None:
      return _registered
    _registered = []
    if not installed():
      return _registered

    try:
      # Registering TensorRT-RTX crashes the process unless the system C++
      # runtime is already loaded.
      ctypes.WinDLL('msvcp140.dll')
      from winui3.microsoft.windows.applicationmodel.dynamicdependency import (
          bootstrap)
      _bootstrap = bootstrap.initialize()
      atexit.register(_bootstrap)
      import onnxruntime as ort
      import winui3.microsoft.windows.ai.machinelearning as winml
      catalog = winml.ExecutionProviderCatalog.get_default()
      # Kept in a list: provider objects can outlive a temporary collection.
      providers = list(catalog.find_all_providers())
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          'Could not start Windows ML, using the CPU: %s. Is the Windows App '
          'SDK Runtime installed?', e)
      return _registered

    for provider in providers:
      name = provider.name
      try:
        if provider.ready_state == winml.ExecutionProviderReadyState.NOT_PRESENT:
          if not download:
            continue
          logging.info('Downloading %s; this may take a while.', name)
        result = provider.ensure_ready_async().get()
        if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
          logging.warning('Could not get %s ready: %s', name, result.status)
          continue
        library_path = str(provider.library_path)
        if not os.path.isabs(library_path):
          # Seen with the experimental WebGPU provider; onnxruntime would
          # look for it in its own folder.
          logging.info('Skipping %s, whose library path is relative.', name)
          continue
        ort.register_execution_provider_library(name, library_path)
        _registered.append(name)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning('Could not register %s: %s', name, e)

    logging.info('Windows ML providers: %s', _registered)
    return _registered


def cache_dir(name: str) -> str:
  """A per-user folder for caches, e.g. TensorRT-RTX's compiled kernels."""
  base = os.environ.get('LOCALAPPDATA') or os.path.expanduser(
      os.path.join('~', 'AppData', 'Local'))
  path = os.path.join(base, 'slippi-ai', name)
  os.makedirs(path, exist_ok=True)
  return path
