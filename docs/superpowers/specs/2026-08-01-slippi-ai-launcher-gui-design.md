# Slippi-AI Launcher GUI — Design

**Date:** 2026-08-01
**Status:** Approved (design phase)
**Owner:** Sean Fay

## Problem

Running Slippi-AI scripts on Windows currently means opening PowerShell, activating the venv, and typing long multi-line commands like:

```powershell
python scripts/eval_two.py `
  --p1.type human `
  --p2.ai.path "C:\Users\seanm\slippi-ai\models\medium-v2" `
  --p2.character FALCO `
  --dolphin.path "C:\...\Slippi Dolphin.exe" `
  --dolphin.iso "C:\...\Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso"
```

The paths, character, and model change enough that hardcoding an alias isn't enough, but retyping the whole command every session is friction. We want a small GUI that makes this a click.

## Goals

- **One-click launch** on Windows: double-click a `.bat`, GUI opens.
- **Cover the common scripts**: `eval_two.py`, `run_dolphin.py`, `run_evaluator.py`, `netplay.py`.
- **Remember settings** between sessions.
- **Zero new Python dependencies** — use `tkinter` from the stdlib.
- **Show live output** and let the user stop the running script.

## Non-goals

- Not a replacement for the CLI — advanced/rare flags don't need first-class UI.
- Not a training/data-pipeline UI (no `train.py`, no dataset tools).
- Not cross-platform-polished — Windows-first; it'll probably run on Linux/macOS but the launcher `.bat` and Ctrl-Break behavior are Windows-specific.
- No automated GUI tests. Manual smoke test + one unit test on command-building.

## User flow

1. Double-click `launcher.bat` in the repo root.
2. `.bat` activates `venv\Scripts\Activate.ps1`, then runs `python scripts\launcher.py`.
3. GUI window opens with the last-used values pre-filled.
4. User picks a tab (default: `eval_two`), tweaks fields, clicks **Run**.
5. The exact command being run is printed to the log panel, then stdout/stderr stream in.
6. User clicks **Stop** or closes the window to end the run.

## UI layout

```
┌─ Slippi-AI Launcher ─────────────────────────────────┐
│ Dolphin: [C:\...\Slippi Dolphin.exe]      [Browse…] │  ← global, shared across tabs
│ ISO:     [C:\...\SSBM.iso]                [Browse…] │
├──────────────────────────────────────────────────────┤
│ [ eval_two ] [ run_dolphin ] [ run_evaluator ] [ netplay ] │
├──────────────────────────────────────────────────────┤
│ ── Player 1 ──────      ── Player 2 ──────           │
│ Type:      [ai ▾]       Type:      [ai ▾]           │
│ Character: [FOX ▾]      Character: [FALCO ▾]        │
│ Model:     [medium-v2 ▾] [Browse…]  (hidden if human)│
│ CPU Level: [ 9 ]        (hidden unless type=cpu)    │
│                                                      │
│ Num games: [ ]  (blank = infinite)                   │
│                                                      │
│ ▸ Advanced (click to expand)                         │
│                                                      │
│ [ Run ]  [ Stop ]        Status: idle                │
├──────────────────────────────────────────────────────┤
│ ┌ Log ────────────────────────────────────────────┐ │
│ │ (streaming stdout/stderr here…)                  │ │
│ │                                                  │ │
│ └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

Global Dolphin/ISO paths sit above the notebook and apply to all tabs. Only one script runs at a time; the log panel is shared.

## Components

### `GlobalPathsFrame`
Two labeled entries + Browse buttons for the Dolphin executable and the ISO. Values are read by every tab that spawns Dolphin.

### `ScriptTab` (abstract base)
Owns:
- A grid of field widgets
- An **Advanced** expander (a `ttk.Labelframe` toggled by a checkbutton) for less-common flags
- **Run** / **Stop** buttons and a status label
- A shared reference to the `LogPanel` and `ProcessRunner`

Subclasses implement `build_argv(global_paths, config) -> list[str]` and `validate() -> list[str]` (returns list of error messages; empty = OK).

Concrete subclasses:
- `EvalTwoTab` — wraps `scripts/eval_two.py`. Uses two `PlayerFrame`s.
- `RunDolphinTab` — wraps `scripts/run_dolphin.py`. One player, plus AI path.
- `RunEvaluatorTab` — wraps `scripts/run_evaluator.py`. Fields TBD by inspecting the script at implementation time; keep the same field-spec pattern.
- `NetplayTab` — wraps `scripts/netplay.py`. Fields TBD similarly.

### `PlayerFrame`
Reusable subwidget for one player. Fields:
- **Type**: `ai` / `human` / `cpu` (dropdown)
- **Character**: dropdown of `melee.Character` names (hardcoded list — see "Character list" below). Hidden when type = `human`.
- **Model**: dropdown listing all immediate subdirectories of `models/` (no filtering — user knows what they put there), plus a Browse button for paths outside `models/`. Hidden when type ≠ `ai`.
- **CPU Level**: integer entry 1–9. Hidden unless type = `cpu`.

Rebinds visibility on type change.

### `LogPanel`
Read-only `tk.Text` + `ttk.Scrollbar`. Colored tags: `stdout` (default), `stderr` (red), `meta` (grey, italic) for lines the launcher itself prints (e.g., "Running: …", "Exit code: 1").

Auto-scrolls to bottom unless the user has scrolled up (detected via yview at write time).

### `ProcessRunner`
Wraps a single `subprocess.Popen`. Only one process at a time.

- `start(argv, cwd)`:
  - `subprocess.Popen(argv, cwd=cwd, stdout=PIPE, stderr=STDOUT, bufsize=1, text=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)`
  - Spawn a daemon reader thread that reads lines and pushes `(kind, line)` tuples onto a `queue.Queue`.
- Tk main loop drains the queue via `root.after(50, drain)` and writes to `LogPanel`.
- `stop()`: send `CTRL_BREAK_EVENT`; if process still alive after 3s, `terminate()`; after another 2s, `kill()`.
- `is_running` property.
- On process exit, pushes a meta line `Exit code: N` and calls a callback so the tab can re-enable Run.

### `Config`
Persisted JSON at `%USERPROFILE%\.slippi_ai_launcher.json`. Schema:

```json
{
  "global": {"dolphin_path": "...", "iso_path": "..."},
  "eval_two":       { ...field values... },
  "run_dolphin":    { ... },
  "run_evaluator":  { ... },
  "netplay":        { ... },
  "last_tab": "eval_two"
}
```

Loaded at startup. Saved on Run (snapshot working configs) and on window close.

## Data flow

1. User clicks **Run** on a tab.
2. Tab calls `validate()`. If errors, show them inline (red label next to each offending field) and abort.
3. Tab calls `build_argv(global_paths, config)` → argv list like:
   ```
   ["python", "scripts/eval_two.py",
    "--p1.type=human",
    "--p2.ai.path=C:\\...\\medium-v2",
    "--p2.character=FALCO",
    "--dolphin.path=C:\\...\\Slippi Dolphin.exe",
    "--dolphin.iso=C:\\...\\SSBM.iso"]
   ```
4. `Config.save()` snapshots all fields.
5. `LogPanel` gets a meta line: `Running: python scripts/eval_two.py --p1.type=human …`
6. `ProcessRunner.start(argv, cwd=repo_root)`. Run disabled, Stop enabled, status → "running".
7. Reader thread → queue → LogPanel.
8. On exit (natural or Stop): Run re-enabled, Stop disabled, status → "exited (code N)".

## Command formatting

Use `--key=value` form (single argv element) rather than `--key value` (two). This avoids ambiguity with paths that start with `-` and matches fancyflags' preferred form.

For nested fancyflags dicts, join with `.`: `--p1.ai.path=…`, `--dolphin.iso=…`.

Boolean flags use `--foo=true` / `--foo=false`.

Blank `num_games` → omit the flag entirely (fancyflags default = infinite).

## Character list

Hardcoded in `launcher.py`:

```python
# From peppi/melee.Character; stable across libmelee versions.
CHARACTERS = [
    "FOX", "FALCO", "MARTH", "SHEIK", "JIGGLYPUFF", "CAPTAIN_FALCON",
    "PEACH", "ICE_CLIMBERS", "PIKACHU", "SAMUS", "DR_MARIO", "YOSHI",
    "LUIGI", "GANONDORF", "MARIO", "YOUNG_LINK", "LINK", "DONKEY_KONG",
    "GAME_AND_WATCH", "MEWTWO", "ROY", "PICHU", "NESS", "BOWSER",
    "KIRBY", "ZELDA",
]
```

Avoids a slow `import melee` at launcher startup. Comment references the source enum in `libmelee`.

## Error handling

| Situation | Behavior |
|---|---|
| Dolphin path or ISO path blank / missing on disk | Red inline label under the field; Run aborted. |
| Model path blank when type=ai | Red inline label; Run aborted. |
| Model path set but doesn't exist | Red inline label; Run aborted. |
| Subprocess fails to start (e.g., `python` not on PATH) | Exception caught, printed to log in red, status → "failed to start". |
| User closes window while script running | Prompt: "Script is running — stop and quit?" If yes, `ProcessRunner.stop()` then `root.destroy()`. |
| `models/` directory empty or missing | Model dropdown shows only "Browse…"; no crash. |
| Config file missing or malformed | Log a warning, start with defaults; overwrite on next save. |

## Testing

- **Unit test** — `tests/test_launcher_command.py`. For each tab class, feed a fixed config dict into `build_argv()` and assert the argv matches an expected list. Locks down flag formatting so refactors don't silently break it. Pure function, no GUI needed.
- **Manual smoke test** — launch, verify that clicking Run with the user's usual settings produces argv byte-identical to their current PowerShell command (shown in the log panel before spawn).

No automated GUI tests. The payoff isn't worth the harness setup for a personal launcher.

## Files touched

- **New** `scripts/launcher.py` — the Tkinter app (~400 lines).
- **New** `launcher.bat` in repo root — activates venv, runs the launcher. ~5 lines.
- **New** `tests/test_launcher_command.py` — unit test on `build_argv()`.

No existing files modified.

## Open questions for implementation

- Exact field list for `run_evaluator.py` and `netplay.py` — determined by reading those scripts during implementation. If any expose radically different fields (not the standard player/dolphin flags), the tab may need custom widgets; call that out in the plan rather than shoe-horning.
- Whether `ProcessRunner` needs to expose the venv's `python.exe` path explicitly instead of relying on the `.bat` having activated it. If launched from an activated venv, `python` resolves correctly; if the user runs `python scripts\launcher.py` from an unactivated shell, subprocesses might use the wrong interpreter. Implementation will use `sys.executable` for the subprocess `python` to guarantee it matches the launcher's own interpreter.
