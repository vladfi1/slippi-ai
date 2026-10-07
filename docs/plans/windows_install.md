# Plan: easy install and play on Windows

Status: Steps 1-2 implemented (2026-10-06): the lean play install with
Windows CI (PR #54, branch `windows-play-install`) and ONNX export with a
jax-free agent (branch `onnx-agent`, stacked on #54). A converted TF model
(`medium-v1`) has been played locally as ONNX in Slippi Dolphin on Windows.
Step 3, GPU inference, is mostly done (2026-10-07, on `onnx-agent`): CUDA
with CUDA graphs, packed graph I/O and float16 weight storage, benchmarked
in the real loop with Dolphin. What's left is measuring low-end machines for
a minimum-spec statement. Step 4, the session library entry point
(`slippi_ai/session.py`), is done (2026-10-07). Step 5, model downloads, is
deferred (client on branch `model-downloads`).

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
  initial state are JSON in the model metadata. Batch size is dynamic or
  fixed (`--batch_size`), and frame skip is supported. Graph inputs and
  outputs are packed by dtype into one `[B, n]` tensor each (layouts in the
  metadata), and weights are stored as float16 by default (see step 3).
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
python -m venv .venv\ort-cuda
.venv\ort-cuda\Scripts\pip install -e ".[onnx-cuda]"  # jax-free play env with CUDA

# TF checkpoint -> JAX -> ONNX (deployed_models/ is gitignored)
.venv\Scripts\python scripts\convert_tf_checkpoint_to_jax.py <tf_model> deployed_models\<name>
.venv\Scripts\python scripts\export_onnx.py --checkpoint deployed_models\<name> --batch_size 1
.venv\Scripts\python tests\onnx_test.py --models <absolute path to deployed_models\name>
.venv\ort-cuda\Scripts\python tests\onnx_providers_test.py --models deployed_models\<name>.onnx

.venv\play\Scripts\python scripts\eval_two.py --p1.type cpu --p2.ai.path deployed_models\<name>.onnx --p2.character fox
.venv\play\Scripts\python scripts\benchmark_eval_two.py --p1.type cpu --p2.ai.path deployed_models\<name>.onnx --p2.character fox
```

Slippi Dolphin is at `%APPDATA%\Slippi Launcher\netplay` and the ISO path is
`isoPath` in `%APPDATA%\Slippi Launcher\Settings`; pass them with
`--dolphin.path`/`--dolphin.iso` or set `DOLPHIN_PATH`/`ISO_PATH`.

Use Python 3.12/3.13 venvs: on Windows a bare `python` may be Python 3.14,
which needs MSVC for pyenet (see below). Installing the export env needs
Windows long paths enabled (orbax ships deeply nested test files).

## 3. GPU inference (mostly done)

Done:

- **CUDA via `onnxruntime-gpu`.** The `onnx-cuda` extra installs
  `onnxruntime-gpu[cuda,cudnn]`, which brings CUDA/cuDNN as pip wheels (no
  separate CUDA install; `ort.preload_dlls()` finds them). It conflicts with
  plain `onnxruntime`, so users install one extra or the other.
- **Provider selection.** Each `OnnxAgent` creates its own session via
  `onnx_policies.SessionRunner`. Agent flags `--p*.ai.onnx.providers`
  (default: CUDA if available, else CPU) and `--p*.ai.onnx.cuda_graph`.
- **No CPU fallbacks on CUDA.** onnxruntime's CUDA provider lacks int32 Clip
  and uint8/uint16 comparison kernels; the export avoids them, and a fixed
  batch size removes the shape arithmetic that onnxruntime places on CPU.
- **CUDA graphs.** A plain CUDA step was ~700 tiny kernel launches (mostly
  game encoding), so launch-bound. With a fixed-batch export, the runner
  binds fixed device buffers (IOBinding) and replays the step as a CUDA graph.
- **Packed I/O.** ~170 graph inputs/outputs became one tensor per dtype,
  cutting host-device copies from ~1.9 ms to near zero.
- **float16 weight storage (default).** Weights are stored as float16 and
  cast back to float32 in the graph; onnxruntime folds the casts at load, so
  compute stays float32. Halves file size (`diamond` 373 -> 187 MB), same
  speed, and no sampled action differed from float32 in 300 frames.
- `tests/onnx_providers_test.py` compares a provider (default CUDA) against
  CPU, or a model against a reference model, with the same inputs each step.

Measured on an RTX 3080 Ti + 16-thread CPU, `agent.step` at batch size 1:

| model | CPU | CUDA, no graph | CUDA graph |
|---|---|---|---|
| medium-v1 (10.6M params) | 1.9-2.5 ms | 6.5 ms | 2.4-2.6 ms |
| diamond (~98M params, custom_v1) | 10.8-12.7 ms | 8.3 ms | 2.6 ms |

Ranges are across runs (CPU timings were noisy with other load). About
0.4 ms of each step is Python (packing, noise, decoding). On this machine the
CPU is enough for `medium-v1`; the GPU matters for large models.

Tried and rejected:

- **DirectML** (`onnxruntime-directml`): slower than CPU for both models
  (30-34 ms/step, p99 up to 62 ms), and its package lags (1.24 vs 1.30).
- **float16 compute**: no faster at batch size 1 (the GPU is overhead-bound),
  slower on CPU, and changed ~10% of `diamond`'s sampled action components
  vs float32 given identical noise. jax2onnx's float16 export produced invalid
  graphs, and onnxconverter-common's converter mistypes some Casts.

**Real loop.** `scripts/benchmark_eval_two.py` runs the `eval_two` loop
(Dolphin at 1x with blocking input, in-game CPU vs the agent) for a fixed
number of frames and reports the achieved fps and per-frame times. On the
same machine, 1800 frames per run:

| model | inference | async | fps | slow frames | agent time/frame (mean / p99) |
|---|---|---|---|---|---|
| medium-v1 | CPU | on | 59.94 | 0 | 1.2 / 1.8 ms |
| medium-v1 | CPU | off | 59.94 | 0 | 3.4 / 4.1 ms |
| medium-v1 | CUDA graph | on | 59.94 | 0 | 1.9 / 4.7 ms |
| medium-v1 | CUDA graph | off | 59.94 | 0 | 9.3 / 11.8 ms |
| diamond | CPU | on | 59.94 | 0 | 1.2 / 1.8 ms |
| diamond | CPU | off | 59.94 | 0 | 12.8 / 14.6 ms |
| diamond | CUDA graph | on | 59.94 | 0 | 1.8 / 3.6 ms |
| diamond | CUDA graph | off | 59.94 | 0 | 7.3 / 14.9 ms |

- Everything keeps up on this machine. Async inference (the `eval_two`
  default) keeps the agent's main-loop time at 1-2 ms, since inference
  overlaps with Dolphin within the online delay.
- CUDA steps are slower in the loop than back to back (2.5 -> 7-9 ms): paced
  at 60 Hz without Dolphin they already take 5.5-6.6 ms (p99 ~13 ms), likely
  because the GPU downclocks between frames; Dolphin's rendering adds the
  rest. CPU inference is barely affected by pacing.

Remaining:

- **Minimum spec.** Run `benchmark_eval_two.py` on low-end machines (e.g. the
  2-core laptop, a 4-core CPU without GPU) to write the README statement
  (e.g. "CPU-only works on 4+ cores; otherwise use a GPU").
- GPU latency when paced: try NVIDIA's "prefer maximum performance" power
  setting, and check whether the async agent hides the p99 spikes on weaker
  CPUs.
- Keep the recurrent state on the device between steps (would save the
  remaining copies; needs ping-pong buffers or a device-side copy, since CUDA
  graphs need fixed addresses).
- If low-end machines still can't keep up, consider a smaller "lite" model;
  phillip's strength is limited by its input delay more than its size.

## 4. Library entry point for a session (done)

`slippi_ai/session.py` holds the `eval_two` loop:

- `SessionConfig(players, dolphin, num_games)`: per-port player flag values
  (`session.player_flags()`) and a `DolphinConfig`
  (`session.default_dolphin_config()`, 1x speed with graphics).
- `Session(config)` starts the agents and Dolphin; `session.frames(stop_event)`
  steps the agents on each in-game frame and yields it with the step time;
  `close()` stops everything.
- `run_session(config, stop_event)` plays with the slow-step warnings.

`scripts/eval_two.py` and `scripts/benchmark_eval_two.py` are thin wrappers.
The stop event (`threading.Event` or `multiprocessing.Event`) is checked
once per frame; setting it from another thread stopped a session and Dolphin
within 0.2 s. It can't interrupt a hung Dolphin, so the GUI should still run
the session in a child process (kill as a fallback), which also keeps
Dolphin/libmelee crashes from taking down the UI.

## 5. Model distribution (deferred)

Deferred (2026-10-07): for now users download model files themselves. A
download client (`--p*.ai.model <name>`, a sha256-pinned manifest shipped with
the package, caching under `%LOCALAPPDATA%\slippi-ai\models`) is implemented
and tested on branch `model-downloads`, waiting on hosting.

- Host exported `.onnx` files (fp32 and/or fp16) on the Hugging Face Hub, or
  GitHub Releases, instead of the Google Drive folder.
- `--p2.ai.model fox` style names that download and cache automatically,
  with the hub repo/revision pinned per slippi-ai release so model format
  changes don't break old installs (the metadata already has a
  `format_version`).
- With steps 4 and 5, a technical user's whole setup is
  `pip install slippi-ai[onnx]` (or `uvx`) plus Dolphin and an ISO.

## 6. Packaging and GUI

Local play only for now; netplay is on the back burner.

- **PyInstaller for the CLI (done, CPU onnxruntime).** `packaging/slippi_ai.spec`
  builds a one-dir bundle with `eval_two.exe` and `benchmark_eval_two.exe`
  sharing one `_internal` (180 MB; pyarrow is 81 MB of it, imported by
  `slippi_ai/types.py`). It collects melee's data files and onnxruntime's
  DLLs, and excludes jax/tf. libmelee's slippstream runs a worker process,
  so the scripts call `multiprocessing.freeze_support()`. Locally the
  frozen `eval_two.exe` played in Slippi Dolphin, and `benchmark_eval_two.exe`
  matched the unfrozen numbers (`medium-v1` CPU: 59.95 fps, 1.1 ms/step).
  `.github/workflows/bundle.yml` builds it on Windows, checks that the exes
  start and that the frozen onnxruntime loads an exported model, and uploads
  the bundle as a workflow artifact for testers.
- Not yet tried: a CUDA bundle (`onnxruntime-gpu` with the pip CUDA/cuDNN
  DLLs, likely >1 GB).
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

- Whether one installer can ship both CPU and CUDA onnxruntime (the CUDA
  wheels are large), and what to offer AMD/Intel GPU users given DirectML's
  results; Windows ML is untested.
- Whether to distribute models exported with a fixed batch size of 1 (needed
  for CUDA graphs) only.
- Whether and when the GUI should also support netplay (`scripts/netplay.py`);
  local play comes first.
