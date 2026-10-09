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
planned as an online model index (2026-10-09) that the GUI browses like local
models; a first client with a packaged manifest is on branch
`model-downloads`.

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
- **Windows ML** (branch `winml`, 2026-10-08, Windows 11 23H2):
  `onnxruntime-windowsml` 1.25.2 plus the `wasdk-` packages. Python only
  supports framework-dependent deployment, so users must install the
  matching Windows App SDK Runtime (2.3.1 here; frameworks other apps had
  installed weren't enough, it needs the Main/DDLM packages; installed
  without admin). The catalog had no providers: vendor ones (TensorRT-RTX,
  MIGraphX, OpenVINO, QNN, VitisAI) need 24H2+, and Windows 10 only gets
  CPU and DirectML. DirectML matched CPU's actions but was slower (median
  14 ms `medium-v1`, 34 ms `diamond` vs CPU 2.0 / 10.8 ms).
  `scripts/winml_benchmark.py` registers catalog providers and times every
  device.
- **Windows ML on 25H2** (same machine, after updating): the catalog offered
  TensorRT-RTX (`NvTensorRTRTXExecutionProvider`, ~13 s first download) and
  WebGPU (experimental; failed to register from Python, its library path is
  relative). TensorRT-RTX rejects uint8 and uint16 tensors and then silently
  leaves the whole graph on CPU; with copies of the models retyped to int32
  (graph I/O, casts and constants) it ran the whole graph, with no action
  differences vs CPU in 320 steps:

  | model | CPU | CUDA graph (onnxruntime-gpu) | TensorRT-RTX (Windows ML) |
  |---|---|---|---|
  | medium-v1 | 2.2 ms | 2.5 ms | 0.7 ms (p99 0.9) |
  | diamond | 11.3 ms | 2.5 ms | 1.2 ms (p99 1.7) |

  Session setup took 4.3 / 7.5 s (engine build, no cache yet).
- **TensorRT-RTX setup caching** (`diamond`, fresh process each time):
  no cache 7.2-8.3 s; runtime cache (`nv_runtime_cache_path`, compiled
  kernels, 1.7 MB) 7.1 s when cold, then 3.2 s; a compiled EP context model
  (`ort.ModelCompiler`, one-time 8.2 s) 4.4 s, or 0.37 s together with the
  runtime cache, but it writes a 394 MB engine (float32 weights, twice the
  float16 `.onnx`). Decided: use the runtime cache only; twice the model
  size on disk per model isn't worth a few seconds of loading.
- Registering TensorRT-RTX from Python segfaults unless the system C++
  runtime (`msvcp140.dll`) is already loaded; load it explicitly first.
- `export_onnx.py --widen_ints` (the default) now exports int32 instead of
  8/16-bit integers: the packed inputs and outputs (3 input tensors instead
  of 5) and, by a graph pass, the casts and constants inside. Real exports
  then run on TensorRT-RTX unmodified (0.7 / 1.2 ms, no action differences
  vs CPU), with CPU and CUDA graph times unchanged, and still match JAX in
  `tests/onnx_test.py`. Older exports keep working; their layouts are in
  the metadata.

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
| medium-v1 | TensorRT-RTX | on | 59.94 | 0 | 1.3 / 2.5 ms |
| medium-v1 | TensorRT-RTX | off | 59.94 | 5 | 5.3 / 11.6 ms |
| diamond | TensorRT-RTX | on | 59.94 | 0 | 1.5 / 2.6 ms |
| diamond | TensorRT-RTX | off | 59.94 | 0 | 6.2 / 9.7 ms |

- Everything keeps up on this machine. Async inference (the `eval_two`
  default) keeps the agent's main-loop time at 1-2 ms, since inference
  overlaps with Dolphin within the online delay.
- CUDA steps are slower in the loop than back to back (2.5 -> 7-9 ms): paced
  at 60 Hz without Dolphin they already take 5.5-6.6 ms (p99 ~13 ms), likely
  because the GPU downclocks between frames; Dolphin's rendering adds the
  rest. CPU inference is barely affected by pacing.
- TensorRT-RTX (Windows ML, 25H2, measured later) behaves the same way:
  0.7-1.2 ms back to back, 5-6 ms paced. With async inference, as in
  `eval_two` and the GUI, both models keep up with no slow frames.

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

## 5. Model distribution (planned)

For now users download model files themselves. Branch `model-downloads`
(2026-10-07) has a first download client: `--p*.ai.model <name>` looks the
name up in a manifest shipped with the package (`slippi_ai/data/models.json`,
built by `scripts/make_model_manifest.py`), downloads the file, checks its
sha256 and caches it by hash. It was never merged, since nothing is hosted.

Decided (2026-10-09): the list of models is hosted online, next to the
files, instead of shipped with the package, so newly published models show
up without a new release. The GUI shows published models alongside local
ones, with the same character, opponent and delay filters, and downloads one
when it's chosen. The packaged manifest is replaced; its download, sha256
check and cache code is reused.

### Hosting (done)

Done (2026-10-09): https://huggingface.co/vladfi/phillip-models, public, with
`diamond` and `falco_d21_ditto_v6.1` (fp16, batch size 1), and a model card
with the MIT license.

- A Hugging Face model repo holds the exported `.onnx`
  files and the index. It's free for public models, serves large files from
  a CDN, keeps every revision, and counts downloads. GitHub Releases would
  work too, but updating the index there means re-uploading release assets.
- The index is one JSON file, read at a fixed URL on the `main` branch:
  `https://huggingface.co/<repo>/resolve/main/index-v1.json`. The schema
  version is in the file name, so an incompatible schema change publishes
  `index-v2.json` next to it, and older installs keep reading (and we keep
  updating, as long as practical) `index-v1.json`.
- Each entry's file URL is pinned to the commit that uploaded it
  (`resolve/<commit>/<file>`), and has its sha256. Overwriting or deleting a
  file on `main` later can't change what an entry downloads, and the hash
  check catches anything else.

### Index format

```json
{
  "updated": "2026-10-09T12:00:00Z",
  "models": [
    {
      "name": "fox-d21",
      "description": "Fox trained with RL against Falco and Fox.",
      "url": "https://huggingface.co/<repo>/resolve/<commit>/fox-d21.onnx",
      "sha256": "...",
      "size": 98041677,
      "format_version": 1,
      "batch_size": 1,
      "agent_type": "RL",
      "characters": ["FOX"],
      "opponents": ["FALCO", "FOX"],
      "delay": 21,
      "published": "2026-10-09"
    }
  ]
}
```

- The entry carries everything the GUI filters and displays without
  downloading: the fields of `eval_lib.AgentSummary` (type, delay,
  characters, opponents) plus `batch_size`, read from the exported model's
  metadata at publish time, the same way `gui/models.scan` reads local
  files. The client builds an `AgentSummary` from an entry, so local and
  remote models go through the same filters.
- Names are unique and stable: a retrained model gets a new name, or the old
  name points at a new file (new sha256, so the cached copy isn't reused).
- Compatibility comes from `format_version` instead of per-release pinning:
  the client hides entries whose `format_version` it can't load
  (`onnx_policies.FORMAT_VERSION`), and, when there are any, says that a
  newer phillip has more models. A format change publishes new files for the
  new version and keeps the old ones (as separate entries) as long as old
  installs matter. Entries also must have `batch_size` 1 to be listed.
- Unknown fields are ignored, so adding fields doesn't need a new schema
  version.

### Client (`slippi_ai/models.py`, done)

Done (2026-10-09), reworked from the `model-downloads` branch; stdlib only
(`urllib`), so the bundle doesn't need `huggingface_hub`. Tested in
`tests/models_test.py` (in `play.yml`) against a local HTTP server, and by
hand: a real exported model served locally downloaded by name and loaded.
Checked against the live index too (fetch, ETag revalidation, download
through the Hub's redirect).

- `fetch_index()` downloads the index (with `If-None-Match`/ETag) and saves
  it to `%LOCALAPPDATA%\phillip\models\index-v1.json`; `load_index()` reads
  that saved copy. Offline, or if the fetch fails, the last saved index is
  used, so the list and downloaded models keep working.
  `$PHILLIP_INDEX_URL` overrides the URL, for tests and staging.
- Downloads go to `%LOCALAPPDATA%\phillip\models\<sha256[:16]>\<file>`
  (named `phillip` instead of the branch's `slippi-ai`, to match the app;
  `$PHILLIP_CACHE` overrides it), written to a `.part` file and renamed after
  the sha256 check, with a progress callback and a cancel flag for the GUI.
- Each download's index entry is saved next to it (`info.json`), and
  `downloaded_models()` lists those, so a model removed from the index stays
  playable, and listable with its metadata, once downloaded.
- `info_from_file()` builds an entry from an exported `.onnx` file, for the
  publishing script.
- CLI: `--p*.ai.model <name>` resolves against the saved index, fetching it
  first if there isn't one or the name is unknown. `<name>@<sha256 prefix>`
  pins an exact file, for reproducible evals. A name can have several
  entries (e.g. one per `format_version`); the compatible one is used.

### GUI (done)

Done (2026-10-09), in `slippi_ai/gui/app.py` and `gui/models.py`. Checked
offscreen against the live index with a fresh cache: listing, filtering,
downloading (with the session start stubbed), cancelling, hiding online
models, deleting a download, and offline with and without a saved index.
Played by hand: `diamond` downloaded with "Download and start" and played in
Slippi Dolphin.

- The model list merges published models (in index order, then downloads
  no longer in the index) and the optional local folder's. A published
  model shows once: as downloaded if that version is, otherwise as online
  (older downloads of it are hidden, and deleted once the new version is
  downloaded). A local file that is also published is listed twice.
- A Status column shows "Downloaded", "Your folder" or "Download (N MB)". A
  checkbox (on by default, saved) hides models that need downloading.
- The models folder is optional ("Your models folder"): a new user can play
  without one.
- For an online model Start reads "Download and start": the download runs
  in a thread with progress in the status line, Start becomes "Cancel
  download", and the controls are locked meanwhile. Right-clicking a
  downloaded model offers "Delete download".
- At startup the saved index and downloads are shown at once, and the index
  is fetched in the background. The "Published models" line shows the
  counts, how many models need a newer phillip, or, offline, "Couldn't
  reach the list of published models; showing the list from <date>" (the
  error is in its tooltip).
- Settings remember a published model as `published:<name>`, so the choice
  carries over to updated versions, and a local one by path (`model`,
  migrated from `model_path`).
- The model's tooltip shows its description and URL or path.

### Publishing (`scripts/publish_models.py`, done)

Replaces `scripts/make_model_manifest.py`; uses `huggingface_hub`, in its own
`publish` extra (publishing only reads the models' metadata, so it doesn't
need jax), and `hf auth login` with a write token.

- The index on the Hub is the list of published models (decided
  2026-10-09): models are published often, and that shouldn't need a commit
  here. The Hub keeps the history, one commit per change.
- `publish_models.py add <file.onnx>... [--name] [--description]` reads each
  model's metadata, uploads files that aren't published yet as
  `<name>.onnx` in one commit, then rewrites `index-v1.json` with their URLs
  pinned to that commit, in a second commit. Re-adding a name replaces its
  entry for that format version (keeping the description unless one is
  given); entries for other format versions stay, for older installs.
- `sync <folder>` (e.g. `onnx_models`) makes the index match a folder:
  `add` for each `.onnx` file in it, named after the file (skipping ones
  that aren't playable), then `remove` for published names with no file.
  Nothing is committed if nothing changed.
- `remove <name>...` drops entries from the index; the files stay, so
  existing downloads and pinned URLs keep working. `describe <name>
  --description=...` edits a description; `list` prints the index.
- `--dry_run` prints the new index and what would be uploaded.

### Steps

1. Client (done): index format, fetch and saved copy, downloads, CLI
   names; tests with a local HTTP server, run in `play.yml`.
2. Publishing script, the Hugging Face repo, and the first two models
   (done).
3. GUI (done): merged list, Download button and progress, offline handling.
4. Bundle (done): `bundle.yml` runs the frozen `eval_two.exe` with
   `--p2.ai.model=falco_d21_ditto_v6.1`, which fetches the live index and
   downloads the model (90 MB) over HTTPS; also checked locally. It uses the
   CLI because the windowed GUI can't run in CI, but the download code is
   the same. It depends on the Hub and on that model staying published.

With sections 4 and 5, a technical user's whole setup is
`pip install slippi-ai[onnx]` (or `uvx`) plus Dolphin and an ISO.

Open: whether to also publish fp32 files (fp16
storage computes in fp32 and halves downloads, so probably not); whether to
show download counts from the Hub in the GUI.

## 6. Packaging and GUI

Local play only for now; netplay is on the back burner.

- **PyInstaller (done).** `packaging/slippi_ai.spec`
  builds a one-dir bundle with the GUI (`phillip.exe`, windowed),
  `eval_two.exe` and `benchmark_eval_two.exe` sharing one `_internal`
  (327 MB with Windows ML, 268 MB with plain onnxruntime: PySide6 is 74 MB,
  pyarrow 81 MB, imported by `slippi_ai/types.py`). It collects melee's data
  files and onnxruntime's DLLs, plus the Windows ML modules and bootstrap DLL
  when built with `.[gui,winml]`, and excludes jax/tf.
  libmelee's slippstream runs a worker process, so the entry points call
  `multiprocessing.freeze_support()`. The GUI's script is a launcher in
  `packaging/gui.py`, because a script's folder goes on the import path and
  `slippi_ai/types.py` would hide the stdlib `types`. Locally the frozen
  `eval_two.exe` played in Slippi Dolphin, and `benchmark_eval_two.exe`
  matched the unfrozen numbers (`medium-v1` CPU: 59.95 fps, 1.1 ms/step).
  `.github/workflows/bundle.yml` builds it with Windows ML, checks that the
  CLI exes start and load an exported model, first without the Windows App
  SDK Runtime (CPU fallback) and then with it (Windows ML starts; the
  runners have no GPU), and uploads the bundle as a workflow artifact for
  testers. The GUI is only
  built there, since a windowed exe reports errors in a dialog.
- **One build, on Windows ML** (decided 2026-10-08): `onnxruntime-windowsml`
  falls back to CPU (and DirectML) on older Windows, including Windows 10,
  and gets TensorRT-RTX on 24H2+ with RTX GPUs, which beat CUDA graphs. No
  separate CPU or CUDA bundles.
  - Done: `pip install .[winml]`. `slippi_ai/winml.py` starts Windows ML once
    per process (preloading msvcp140) and registers the catalog's providers,
    downloading them if needed. `onnx_policies.default_providers()` prefers
    TensorRT-RTX, then CUDA, then CPU; `SessionRunner` creates TensorRT-RTX
    sessions by device (it fails by name, though onnxruntime lists it as
    available), with its runtime cache in
    `%LOCALAPPDATA%\phillip\tensorrt-rtx` (written when the session
    closes; `diamond` 6.5 s, then 3.5 s), and falls back to CPU if that
    fails.
  - Done since: the real-loop benchmark, a "Run on" choice in the GUI, and
    the bundle. The frozen `eval_two.exe` ran `diamond` on TensorRT-RTX; the
    bundle carries its own `msvcp140.dll` (14.50, as in System32).
  - The installer installs the Windows App SDK Runtime (2.3.1) when it's
    missing; see Installer. Without it the app falls back to the CPU
    (checked in CI, not yet locally), so the zipped bundle works as is.
- **GUI: PySide6 (done, local play).** `python -m slippi_ai.gui`
  (`pip install slippi-ai[gui,onnx]`) or the bundle's `phillip.exe`.
  - One window, with settings saved to `%APPDATA%\phillip\gui.json`.
    Slippi Dolphin and the ISO are found from Slippi Launcher's settings
    (Browse otherwise); the ISO is MD5-checked against NTSC 1.02.
  - phillip's models come from a folder of `.onnx` files (searched
    recursively). The user picks phillip's character and optionally the
    opponent's, and chooses among the models that play that matchup; the
    list shows each model's reaction delay and who it was trained against.
    Only the metadata is read (`onnx_policies.read_metadata_from_file` skips
    the graph in the protobuf), about 1 ms per model instead of up to 0.8 s
    with an onnxruntime session.
  - Only models exported with batch size 1 are listed (decided 2026-10-08):
    play runs one game, CUDA graphs need a fixed size, and that's all we
    distribute for the GUI; `export_onnx.py` defaults to it. Power users who
    want to run evals can download the original TF/JAX checkpoints.
  - The opponent is either the user, in a chosen port, or an in-game CPU.
  - A checkbox (on by default) starts Dolphin with a copy of Slippi Dolphin's
    settings (`copy_home_directory`): graphics, audio, and a human's
    controller config. Otherwise Dolphin uses defaults and a human's port is
    a GameCube adapter.
  - Start/Stop and a log panel. The session runs in a spawn-context child
    process (not a daemon, since slippstream starts its own worker) and is
    killed if it doesn't stop within 10 s. Start warns if Dolphin is already
    running.
  - Tested against an in-game CPU and with a controller.
  - Published models are listed and downloaded on demand (step 5).
  - Not yet: a Wii U adapter driver check, netplay.
- **Installer: Inno Setup (started).** `packaging/slippi_ai.iss` wraps the
  PyInstaller output in `dist/phillip-setup-<version>.exe` (80 MB).
  - Installs per user by default, to `%LOCALAPPDATA%\Programs\phillip` with
    no admin (all users is offered), with a Start menu shortcut and an
    optional desktop one. Upgrades replace `_internal`; uninstalling removes
    `%LOCALAPPDATA%\phillip` (downloaded models and updates, the TensorRT-RTX
    cache) but keeps the GUI's settings.
  - If the Windows App SDK Runtime 2.3.1+ isn't installed (checked with
    `Get-AppxPackage`), a checked-by-default task downloads Microsoft's
    installer (113 MB, SHA-256 pinned) on the Ready page and runs it with
    `--quiet` after copying the files. A failed download only warns: the
    app runs on the CPU. Downloading keeps the setup small and skips the
    runtime for most players who already have it.
  - Tested locally: a silent per-user install (with the runtime check forced
    to fail, so it downloaded and ran the runtime installer), the installed
    `eval_two.exe` on TensorRT-RTX, and a silent uninstall. CI builds the
    installer, installs it silently on a runner without the runtime, and
    runs the installed `eval_two.exe` with Windows ML.
  - To do: try the wizard by hand. Code signing (e.g. Azure Trusted
    Signing) to avoid SmartScreen and antivirus false positives; otherwise
    document the "More info > Run anyway" step. An icon.
- **App version and releases (done, 2026-10-09).** The app (GUI, bundle and
  installer) has its own version, `VERSION` in `slippi_ai/gui/version.py`,
  independent of slippi-ai's in `setup.cfg`; it continues from the 0.2.0
  installer at 0.3.0. Releases are tagged `launcher-v<version>`; pushing one
  makes `bundle.yml` check it matches `VERSION`, build the installer and
  publish it as a GitHub release (`gh release create --generate-notes`).
  The bundle also records the slippi-ai version and git commit it was built
  from (`build_info.json`), which the GUI writes at the top of each session's
  log, for bug reports.
- **In-app updates (done, 2026-10-09).** `slippi_ai/gui/updates.py`: the
  installed app checks GitHub's releases at startup for a newer
  `launcher-v*` release (not drafts or prereleases), and shows a bar with
  "Update", "Later" and a "What's new" link. Update downloads the installer
  (with progress, cancellable) and checks it against the sha256 GitHub
  publishes for release assets; without one, the button opens the release
  page instead. It then runs the installer with `/SILENT /relaunch=1` and
  closes. The app holds a mutex (`phillip-launcher`); setup waits up to 30
  s for it to be released (then asks the user to close phillip), upgrades
  in place and, with `/relaunch=1`, starts the app again. A file the app
  downloads isn't marked as from the internet, so SmartScreen doesn't flag
  the unsigned installer. Only the frozen app checks
  (`$PHILLIP_CHECK_UPDATES=1` forces it, `0` disables it), and not during a
  session or model download.
  - Tested: `tests/updates_test.py` (in `play.yml`); the GUI offscreen
    against a fake releases server (bar, download, cancel, installer
    started, window closed); and the real installers, with a test AppId
    (`/DAppId=...`): 0.3.0 installed, then the 0.3.1 installer started by
    `updates.run_installer` waited while the mutex was held, upgraded once
    it was released and relaunched the app.
  - 0.2.0 installs have no update check; their users install 0.3.0 by hand.
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

- How Windows ML's AMD and Intel providers (MIGraphX, OpenVINO) compare
  with CPU; needs testers with that hardware on 24H2+.
- Whether and when the GUI should also support netplay (`scripts/netplay.py`);
  local play comes first.
