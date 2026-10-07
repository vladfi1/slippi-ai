# Plan: easy install and play on Windows

Status: Steps 1-2 implemented (2026-10-06): the lean play install with
Windows CI (PR #54, branch `windows-play-install`) and ONNX export with a
jax-free agent (branch `onnx-agent`, stacked on #54). A converted TF model
(`medium-v1`) has been played locally as ONNX in Slippi Dolphin on Windows.
Next is step 3, GPU inference, which needs a machine with a real GPU.

## Goal

A Windows player with no Python experience downloads an installer, points it
at their Melee ISO, picks a character and plays phillip, using their GPU when
they have one. Technical users can still `pip install` and use the CLI.

The project is developed on macOS/Linux, so every step should be verifiable
in CI on Windows (`.github/workflows/play.yml`), with occasional manual
testing on real Windows machines with Dolphin and controllers.

## Done

### 1. Lean play install (PR #54)

- `install_requires` is only what's needed to play; training dependencies
  (wandb, pandas, peppi-py, py7zr, fsspec, ...) moved to the `train` extra.
- `py7zr`/`fsspec` are imported lazily in `slippi_db/utils.py` and
  `slippi_ai/data.py`, and wandb lazily in `slippi_ai/tf/train_lib.py`
  (loading TF checkpoints imports it).
- `tests/play_imports_test.py` and `tests/play_agent_test.py` check that the
  play path doesn't import training-only modules. `play.yml` runs them on
  Windows/macOS/Linux x Python 3.12/3.13.

### 2. ONNX export and jax-free agent (branch `onnx-agent`)

- `scripts/export_onnx.py` (`slippi_ai/jax/onnx_export.py`) exports a JAX
  checkpoint to one `.onnx` file doing a single agent step, including game
  encoding. The graph outputs encoded actions, which the agent decodes with
  numpy (`onnx_policies.ControllerDecoder`, from the controller config in the
  metadata), so `custom_v1` models work too. Sampling noise is a graph input:
  `jax.random.categorical`/`bernoulli` are replaced during tracing by
  `argmax(logits + gumbel(u))` and `u < p`. Config, name map, agent config and
  initial state are JSON in the model metadata. Batch size is dynamic and
  frame skip is supported.
- `slippi_ai/onnx_policies.py` runs exported models with numpy and
  onnxruntime as `Platform.ONNX`. `saving.load_state_from_disk` accepts
  `.onnx` paths, so `eval_two.py --p2.ai.path=model.onnx` works.
- `tests/onnx_test.py` replays the OnnxAgent's noise through the JAX step
  function and checks that sampled controllers match exactly and recurrent
  states within tolerance. `play.yml` also exports a model and runs it in a
  venv with only `.[onnx]`.
- TF checkpoints are converted first with
  `scripts/convert_tf_checkpoint_to_jax.py` (fixed for frame-skip policies).
- Known difference: one-hot game inputs with the `ERROR` policy are clamped
  in the graph instead of raising.

Measured on a 2-core i7-5500U laptop (no discrete GPU), `medium-v1`
(10.6M params, delay 21): ONNX fp32 CPU ~14 ms/step (~11.5 ms of it in
`session.run`), JAX fp32 CPU ~49 ms/step, against a 16.7 ms frame budget.
The model plays, but the machine struggles to run phillip and Dolphin
together.

## Resuming on another machine

```
git fetch && git checkout onnx-agent
python -m venv .venv
.venv\Scripts\pip install -e ".[tf,onnx-export]"   # export/convert env
python -m venv .venv\play
.venv\play\Scripts\pip install ".[onnx]"           # jax-free play env

# TF checkpoint -> JAX -> ONNX
.venv\Scripts\python scripts\convert_tf_checkpoint_to_jax.py <tf_model> jax_models\<name>
.venv\Scripts\python scripts\export_onnx.py --checkpoint jax_models\<name>
.venv\Scripts\python tests\onnx_test.py --models <absolute path to jax_models\name>

.venv\play\Scripts\python scripts\eval_two.py --p1.type cpu --p2.ai.path jax_models\<name>.onnx --p2.character fox
```

Use the venv's python: on Windows a bare `python` may be the Microsoft Store
Python 3.14, which has no project dependencies (and see the pyenet issue
below).

## 3. GPU inference (next)

The agent currently uses whatever providers `onnxruntime` exposes, which for
the plain `onnxruntime` package is CPU only.

- **Provider selection.** Add an `onnx.providers` agent flag
  (`eval_lib.BATCH_AGENT_FLAGS['onnx']`), plumbed to `OnnxPolicy` (today
  `saving.load_policy_from_state` creates the policy before agent kwargs are
  applied, so the flag needs to reach `onnx_policies.load_policy_from_state`).
  Default: the best available provider, falling back to CPU.
- **Which runtime package.** Candidates: `onnxruntime-directml` (any DX12 GPU:
  NVIDIA/AMD/Intel, no CUDA install), `onnxruntime-gpu` (CUDA, NVIDIA only,
  needs matching CUDA/cuDNN, possibly via pip `nvidia-*` wheels), and the
  newer Windows ML packaging. These packages conflict with `onnxruntime`, so
  the extras need care, e.g. `onnx-directml`/`onnx-cuda` extras instead of
  adding to `onnx`. Check DirectML's maintenance status before committing.
- **fp16 export.** Add `--dtype float16` to `export_onnx.py` (cast the policy
  in `export_state` before tracing; keep inputs/outputs and sampling in
  fp32), and a looser-tolerance mode in `tests/onnx_test.py`. Only worth it on
  GPU; onnxruntime's CPU kernels mostly lack fp16. For reference, JAX agents
  in `eval_two` default to fp16 (`--p*.ai.jax.dtype`).
- **Per-call overhead.** At batch size 1 a GPU step is dominated by launch and
  transfer overhead; measure before assuming a win. If inputs/outputs
  dominate, use IOBinding to keep the recurrent state on the device, and
  consider packing the ~150 game inputs into fewer tensors.
- **Benchmark the real loop**, not just inference: `eval_two` with Dolphin
  running, with and without `--p2.ai.async_inference`, on CPU and GPU. The
  goal is a minimum-spec statement for the README (e.g. "CPU-only works on
  4+ cores; otherwise use a GPU").
- If low-end machines still can't keep up, consider a smaller "lite" model;
  phillip's strength is limited by its input delay more than its size.

## 4. Library entry point for a session

Refactor `scripts/eval_two.py` into something like
`run_session(config: SessionConfig, stop_event) -> None` in a module, with
the absl script and the future GUI as thin wrappers. The GUI should run the
session in a child process so that Stop is reliable and Dolphin/libmelee
crashes don't take down the UI.

## 5. Model distribution

- Host exported `.onnx` files (fp32 and/or fp16) on the Hugging Face Hub, or
  GitHub Releases, instead of the Google Drive folder.
- `--p2.ai.model fox` style names that download and cache automatically,
  with the hub repo/revision pinned per slippi-ai release so model format
  changes don't break old installs (the metadata already has a
  `format_version`).
- With steps 4 and 5, a technical user's whole setup is
  `pip install slippi-ai[onnx]` (or `uvx`) plus Dolphin and an ISO.

## 6. Packaging and GUI

- **PyInstaller first, for the CLI**, built in a Windows GitHub Actions job,
  to solve bundling issues (onnxruntime DLLs, melee package data, absl/
  fancyflags) before any UI exists. One-dir mode.
- **GUI: PySide6.** First-run setup screen: find Slippi Dolphin (under
  `%APPDATA%`, with a Browse fallback), pick and hash-check the ISO (it can't
  be shipped), warn about Wii U adapter drivers (Zadig/WinUSB) and running
  Dolphin instances. Main screen: character/model picker with download
  progress, opponent port and controller type, Start/Stop, log panel.
- **Installer:** Inno Setup around the PyInstaller output, published on
  GitHub Releases. Code signing (e.g. Azure Trusted Signing) to avoid
  SmartScreen and antivirus false positives; otherwise document the
  "More info > Run anyway" step. A startup check against GitHub Releases for
  new versions.
- Recruit Windows testers from Discord once a packaged build exists.

## Known Windows issues

- `pyenet-vladfi` (a `melee` dependency) has no cp314 Windows wheels, so
  installing on Python 3.14 (the current python.org default) requires MSVC.
  Fix: publish cp314 wheels; until then, document Python 3.12/3.13.
- Training data loaders and the sim env use the `forkserver` multiprocessing
  context, which doesn't exist on Windows (`slippi_ai/data.py`,
  `datasets.py`, `envs.py`). Play is unaffected; training on Windows would
  need a `spawn` fallback.
- Windows `tf-nightly` wheels lag months behind `tfp-nightly`; another reason
  to keep TF off the play path.
- On Windows a running Dolphin instance can block the bot's inputs (already
  noted in the README).

## Open questions

- Which GPU runtime to default to on Windows (DirectML vs CUDA vs Windows
  ML), and whether one installer can ship both.
- fp32 vs fp16 as the distributed default, and whether to ship both.
- Whether the GUI should also support netplay (`scripts/netplay.py`) or only
  local play at first.
