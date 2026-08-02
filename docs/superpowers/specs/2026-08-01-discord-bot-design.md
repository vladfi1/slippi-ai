# Slippi-AI Discord Bot — Design

**Date:** 2026-08-01
**Status:** Approved (design phase)
**Owner:** Sean Fay
**Depends on:** `docs/superpowers/specs/2026-08-01-slippi-ai-launcher-gui-design.md` (adds a tab to that launcher)

## Problem

The launcher can already run one netplay match at a time on Sean's machine. Sean's small Discord server would like to challenge the bot without Sean having to type each match's connect code and character into the GUI. He wants a Discord command his friends can invoke — `/play code:TNBN#217 char:falco` — that spawns the same `netplay.py` subprocess the launcher already builds.

## Goals

- **In-tab UX.** The bot lives as a new tab in the existing Tkinter launcher; start/stop is a button, not a separate script.
- **One match at a time.** Reject overlapping requests; no queue.
- **No new user setup for players.** They only need to know the slash command.
- **Reuses the launcher's existing plumbing.** Same `ProcessRunner`, same `LogPanel`, same argv builder as `NetplayTab`.
- **Fires up occasionally.** Not a 24/7 service. Sean turns it on when friends are around.

## Non-goals

- Not a match-queueing system (rejected in favor of "busy" replies).
- Not a leaderboard, match history, or replay uploader.
- Not model-picking from Discord (the bot uses whichever model Sean loaded in the tab at bot-start time).
- Not multi-model or multi-instance (one Dolphin at a time).
- Not auto-restart / crash recovery (Sean is at the keyboard when it's running).

## User flow

1. Sean opens the launcher, switches to the **Discord** tab.
2. Fills in: bot token, allowed channel IDs, model file, `user.json` path, supported characters (checkboxes).
3. Clicks **Start bot**. Status label shows "connected as `SlippiAIBot#1234`".
4. In a Discord channel from the allowed list, a friend runs `/play code:TNBN#217 char:falco`. Autocomplete on `char` shows only the checked characters.
5. Bot posts: **@friend vs `TNBN#217` as `falco` — starting…**.
6. Netplay spawns (same as if Sean had clicked Run on the Netplay tab). Dolphin window appears on Sean's screen.
7. When netplay prints its `[NETPLAY_MATCH_STARTED]` marker, bot posts **match live**.
8. Match ends (natural exit, timeout before connect, or Sean hits Stop). Bot posts **match ended (exit code 0)** — or, on failure, the last 10 stderr lines in a code block.
9. Slot is free; the next `/play` request can be accepted.
10. Sean clicks **Stop bot** when done for the night.

## UI layout (new Discord tab)

```
┌─ Discord ────────────────────────────────────────────┐
│ Bot token:            [•••••••••••••••••••••]        │
│ Allowed channel IDs:  [1234567890, 9876543210]        │
│ Model path:           [C:\...\models\medium-v2] [Browse…] [Use netplay tab's] │
│ Slippi user.json:     [C:\...\user.json]      [Browse…]  │
│ Supported characters: ☑ fox  ☑ falco  ☐ marth  ☐ sheik  … │
│ Connect timeout (s):  [600]  (only until match starts)  │
│                                                        │
│ [ Start bot ]  [ Stop bot ]     Status: stopped        │
│ Match slot: idle                                       │
└────────────────────────────────────────────────────────┘
```

Log panel and global paths (Dolphin/ISO) at the top of the window are shared with all tabs, as usual.

## Components

### `DiscordTab(ScriptTab)`

Subclass of the existing `ScriptTab`, but with `Run` renamed to **Start bot** and `Stop` to **Stop bot**. `build_argv`/`validate` are unused in the parent-class sense; instead the tab owns a `DiscordBotThread` and delegates start/stop to it.

- `_build_widgets()` — fields above.
- `_values()` — snapshot for `Config` persistence.
- `_on_start()` — validate, instantiate `DiscordBotThread`, call `start(config_dict)`.
- `_on_stop()` — `thread.stop()`, wait up to 5 s in a non-blocking `after`-poll (same pattern as `ProcessRunner.stop`).
- `_status(text)` — thread-safe; the bot thread calls it via `self.app.root.after(0, ...)`.

Adds a second status label: "Match slot: idle / running (@user vs code)".

### `DiscordBotThread`

Daemon thread with its own `asyncio` loop. Owns:
- `discord.Client(intents=discord.Intents.default())`
- The `app_commands.CommandTree` with `/play` registered
- `self._active: MatchRequest | None` (single-slot guard, mutated only inside the loop)
- A reference to the launcher's `ProcessRunner` and `LogPanel` (borrowed, not owned)

Exposes:
- `start(config: dict) -> None` — must be called from the Tk main thread; spins the daemon thread.
- `stop() -> None` — signals loop to close client and exit; safe from Tk main thread.
- `is_running: bool`

Internal:
- `_run()` — thread entry; creates the loop, calls `client.start(token)`, tears down on exit.
- `_on_ready()` — updates status via `root.after(0, tab._status, 'connected as X')`.
- `_on_play(interaction, code, char)` — the slash-command handler; runs entirely in the discord thread's loop.
- `_watch_stdout(line)` — installed as a `LogPanel.append` observer. Scans for `[NETPLAY_MATCH_STARTED]`; when seen, cancels the connect-timeout task and posts "match live".
- `_on_netplay_exit(code)` — installed as the `ProcessRunner.on_exit` callback; posts the end message and clears `_active`. Called on the Tk main thread by `ProcessRunner._drain`; uses `client.loop.call_soon_threadsafe` to bounce back to the discord thread for posting.

### `MatchRequest`

```python
@dataclasses.dataclass
class MatchRequest:
  user_id: int
  user_name: str          # for logs / status
  channel_id: int
  connect_code: str
  character: str          # lowercase libmelee name
  started_at: float       # time.monotonic()
```

### Argv construction

Extract `NetplayTab.build_argv` into a module-level helper `build_netplay_argv(global_paths, tab_values)` so both `NetplayTab` and `DiscordBotThread._on_play` call it. Signature unchanged.

The Discord tab constructs `tab_values` per-request:
```python
tab_values = {
    'model_path': self.model_path,
    'char': request.character,
    'costume': '',
    'connect_code': request.connect_code,
    'runtime': '',
    'user_json_path': self.user_json_path,
}
```

Global paths come from `self.app.global_paths.values()` (Dolphin path + ISO), same as any other tab.

### Marker in `netplay.py`

Add exactly one line, immediately after the `agent.step(gamestate)` inside the game-started branch (after `num_frames = 1`):

```python
print('[NETPLAY_MATCH_STARTED]', flush=True)
```

`flush=True` is required — Python buffers stdout when piped to a subprocess, and the bot needs the marker in real time.

## Slash command

```
/play code:<string, required> char:<choice, required>
```

- `code`: regex-validated `^[A-Z0-9]{2,6}#\d{1,6}$` (Slippi connect-code shape). If it fails, ephemeral reply "Invalid code — expected `ABCD#123`." and abort.
- `char`: `app_commands.Choice`-typed. The choice list is a **snapshot** of the checked characters taken at bot-startup time (Discord slash commands don't support live-updated choices). Changing the checkboxes while the bot is running has no effect until Stop + Start. Defense-in-depth: `_on_play` re-validates against the current live checkbox state, so unchecking a character mid-session immediately stops new matches for it (with an ephemeral "Model doesn't support that character" reply) even though the autocomplete may still offer it until restart.
- Channel allowlist enforcement happens before code/char validation. If the channel isn't allowed, silently do nothing (avoids revealing the bot in non-allowed channels).
- Slash commands are registered per-guild at bot-startup. On start, the bot enumerates the guilds it's in and syncs the command tree to each. On stop, no cleanup — the command persists in Discord's cache but returns "bot is offline" when invoked.

## Data flow

1. `_on_play` fires in the discord loop.
2. Check channel allowlist → check `_active is None` → validate code → validate char.
3. If busy: post `"Bot busy — @<other user> is playing vs <code>."` and return.
4. Set `self._active = MatchRequest(...)`, post start message, note the `interaction.channel` for future posts.
5. Build argv (`build_netplay_argv`). If the argv builder or validation raises, ephemeral-reply the error and clear `_active`.
6. Call `self._process_runner.start(argv, cwd=REPO_ROOT, on_exit=self._on_netplay_exit_scheduled)`. This runs on the discord thread — but `ProcessRunner.start` only touches its own state and spawns a subprocess; the internal `after`-poll is scheduled from the discord thread via `self._root.after`, which is safe because Tk's `after` is documented thread-safe on Windows.
7. Schedule the connect-timeout: `self._timeout_task = asyncio.create_task(self._connect_timeout())`. Sleeps for `connect_timeout_s`, then if the marker was never seen, calls `self._process_runner.stop()` and posts "opponent never joined — timing out".
8. `LogPanel.append` is invoked from the Tk main thread (via `ProcessRunner._drain`). We install a callback: `self._log.on_line = self._watch_stdout`. `_watch_stdout` runs on the Tk main thread; if it sees the marker, it schedules `self._on_match_started()` on the discord loop via `client.loop.call_soon_threadsafe`.
9. `_on_match_started` cancels `_timeout_task` and posts "match live".
10. When the subprocess exits, `ProcessRunner._drain` invokes `on_exit(code)` on the Tk main thread. `_on_netplay_exit_scheduled` uses `call_soon_threadsafe` to post the end message from the discord loop. If exit code is non-zero, includes the last 10 lines from a small stderr ring buffer (added to `LogPanel` — a `collections.deque(maxlen=20)` of recent lines).

## Threading model

Two threads share the launcher:

- **Tk main thread** — runs the GUI. Owns `LogPanel`, `ProcessRunner`, `Config`, all widgets. Handles the subprocess-drain `after` chain.
- **Discord thread** — runs the `asyncio` loop. Owns the `discord.Client`, `_active`, `_timeout_task`.

Cross-thread coordination:
- **Discord → Tk:** never call Tk methods directly. Use `self.app.root.after(0, callable, *args)`. Only used for status label updates and (potentially) posting an appended line to the log panel.
- **Tk → Discord:** use `self._client.loop.call_soon_threadsafe(callable, *args)`. Only used to schedule "match started" and "match ended" reactions after Tk sees the stdout/exit.

`self._active` is only read/written from the discord thread, so no lock is needed.

## Error handling

| Situation | Behavior |
|---|---|
| Bot token invalid | `client.start` raises `LoginFailure`. Thread catches, calls `_status('bad token')`, exits. Start button re-enabled. |
| Discord network drop | `discord.py` auto-reconnects. `on_disconnect` → status "reconnecting…"; `on_resumed` → status "connected as X". |
| `/play` while a match is running | Public reply: "Bot busy — @X is in a match against `code`." |
| Bad connect code | Ephemeral reply: "Invalid code — expected `ABCD#123`." |
| Character not in supported list | Autocomplete prevents; defense-in-depth ephemeral reply "Model doesn't support that character." |
| `netplay.py` fails to spawn (bad path etc.) | Bot posts "failed to start" + last 10 stderr lines. `_active` cleared. |
| Opponent never joins (no marker within timeout) | `ProcessRunner.stop()` (non-blocking cascade already exists), bot posts "opponent never joined — timing out". |
| Match runs indefinitely (opponent joined) | No timeout — runs until Dolphin exits or Sean clicks Stop bot / closes launcher. |
| User closes launcher window | Existing `_on_close` prompts; extend it so the bot thread also gets `stop()` called before `destroy()`. |
| Bot started with fields missing | `validate()` before `_on_start()` returns errors; inline red label; bot not started. |

## Security

- **Token storage:** `%USERPROFILE%\.slippi_ai_launcher.json`. Plaintext. Add `.slippi_ai_launcher.json` to `.gitignore` even though it lives outside the repo — defense in depth in case a future refactor moves it.
- **Subprocess args:** `subprocess.Popen(list, shell=False)` (already how `ProcessRunner` works). `code` is regex-validated; `char` is a fixed choice; `user_json_path` and `model_path` are Sean-controlled fields, not user-supplied.
- **Channel allowlist:** silent-drop for non-allowed channels. Prevents the bot from being usable in random servers if invited.
- **No DMs:** the bot ignores DMs entirely (channel-ID check fails). No "friend just DMs the bot" flow.
- **Rate limits:** none in v1. Small friend server; overlap prevention is enough. If abuse ever happens, add a per-user cooldown.

## Testing

- **Unit tests** (added to `tests/test_launcher_command.py`):
  - `build_netplay_argv` extracted helper — same as existing `NetplayTab.build_argv` tests, but tests both entry points share the code path.
  - `validate_connect_code('TNBN#217')` returns True; various malformed inputs return False.
  - `validate_supported_character('fox', ['fox', 'falco'])` returns True; unsupported returns False.
- **No live Discord tests.** `discord.py` isn't unit-testable without a real gateway connection. Manual smoke test only:
  1. Create a Discord bot at discord.dev, invite it to a test server, copy token + channel ID.
  2. Start the bot; run `/play` against your own connect code.
  3. Confirm start message, "match live" appears when marker fires, and end message appears on exit.
  4. Test the "bot busy" reject by running `/play` twice.
  5. Test the timeout by running `/play` with a made-up code that no-one will join.

## Files touched

- **Modify** `scripts/launcher.py` — new `DiscordTab`, `DiscordBotThread`, `MatchRequest`, extract `build_netplay_argv` as a module-level helper (~250 lines added).
- **Modify** `scripts/netplay.py` — one `print('[NETPLAY_MATCH_STARTED]', flush=True)` line.
- **Modify** `tests/test_launcher_command.py` — validator tests, argv-builder shared path.
- **Modify** `requirements.txt` — add `discord.py>=2.3`.
- **Modify** `.gitignore` — add `.slippi_ai_launcher.json`.

## Open questions

None load-bearing. Two minor decisions deferred to implementation:

- **Exact allowed-channel input format:** comma-separated in one entry vs. one entry per row. Going with comma-separated (simpler UI, matches "small friend server" scale).
- **Whether to also mirror bot events into `LogPanel`:** yes, prefix with `[discord]`. Free debugging value.
