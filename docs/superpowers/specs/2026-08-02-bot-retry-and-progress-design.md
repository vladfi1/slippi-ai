# Discord Bot — Retry & Progress Updates Design

**Date:** 2026-08-02
**Status:** Approved (design phase)
**Owner:** Sean Fay
**Depends on:** `docs/superpowers/specs/2026-08-01-discord-bot-design.md` (extends the existing DiscordBotThread)

## Problem

Two persistent friction points in the current bot:

1. **Menu-helper flakiness.** libmelee occasionally lands in the wrong netplay menu (e.g. Team Battle instead of Direct Connect), causing the match to stall until the connect-timeout fires. Users see a silent 10-minute wait then a "timing out" message.
2. **No progressive feedback.** Between "starting…" and "match live" (or "timing out"), the requester has no idea whether anything is happening. If the bot silently retries, they need to see that too.

## Goals

- Automatically retry pre-match failures (wrong menu, opponent no-show) up to N times without user intervention.
- Post concise progress updates so the requester always knows the bot's state — no more silent waits.
- Keep every added message *load-bearing*: no heartbeats, no menu-state chatter, no "Dolphin launched" ceremony.

## Non-goals

- Not diagnosing *why* libmelee picked the wrong menu (that's an upstream libmelee/slippi-ai bug; the retry side-steps it).
- Not retrying mid-match failures (if the match started and then crashed, that's a real error worth surfacing).
- Not a queue for multiple simultaneous requests (single-slot behavior stays).
- No per-menu-state progress messages ("entered CSS", "picked character", etc.). Only transitions with real information get posts.

## Key insight

Wrong-menu detection *is* retry-on-timeout. Any pre-match failure mode — wrong menu, hung helper, opponent no-show — presents the same way to the bot: netplay never prints `[NETPLAY_MATCH_STARTED]`. So the same retry logic handles all three. No libmelee menu-state introspection needed.

## User flow

1. Requester posts `@bot ANDU#504 falco`.
2. Bot posts the existing starting message (unchanged): `**@you** vs ANDU#504 as falco — starting…\nEnter my code in your Slippi game: **`CHUD#953`**`
3. Netplay attempt 1 runs. Suppose the menu-helper picks the wrong menu.
4. Connect timeout fires without `[NETPLAY_MATCH_STARTED]` marker.
5. Bot kills the netplay process, waits 2 s, posts: `Opponent didn't join, retrying (2/2)…`
6. Netplay attempt 2 runs. Suppose it works this time.
7. Marker fires. Bot posts existing `**match live**`.
8. Match plays out normally. On exit, bot posts: `Match ended after 4m 12s.`

If both attempts fail: `Opponent didn't join after 2 attempts.` No stderr tail (the local log panel has details).

## Configuration

New field on the Discord tab, persisted to config:
- **"Max attempts per request"** — integer, default `2`, range `1–5`. Value of `1` disables retries entirely.

Existing "Connect timeout (s)" field is unchanged — it caps *each attempt*.

## Components

Everything lives in `scripts/launcher.py` (extending `DiscordBotThread`). No new files.

### `MatchRequest` — extended

Add three fields to the existing dataclass:

```python
attempt: int = 1            # 1..max_attempts
max_attempts: int = 2
match_started: bool = False # flipped True on the marker
```

Old callers (currently none other than the bot itself) get defaults.

### `DiscordBotThread` — new / changed methods

- **`_handle_play`** — unchanged through the "starting…" message. Instead of directly spawning netplay and creating a single timeout task, it now calls:
  ```python
  await self._start_attempt(request, channel, first=True)
  ```
- **`_start_attempt(request, channel, first: bool) -> None`** — new. If `first=False`, posts the retry message. Then does what the current spawn code does (installs log watcher, calls `runner.start`, creates connect-timeout task). Stores `request` and `channel` on `self._active` / an instance attribute so the exit and timeout handlers can find them.
- **`_on_match_started`** — sets `self._active.match_started = True`, cancels the connect timeout, posts `**match live**` (unchanged behavior).
- **`_connect_timeout`** — no longer directly posts "opponent never joined". Instead calls `_end_attempt(reason='timeout')`.
- **`on_exit_tk`** (closure inside `_start_attempt`) — schedules `_end_attempt(reason='exit', exit_code=…, recent_lines=…)` on the bot loop.
- **`_end_attempt(reason, exit_code=None, recent_lines=None)`** — new. Central decision point:
  - Cancel `_timeout_task` if still active (idempotent — may have already been cancelled).
  - If `reason == 'timeout'` AND the netplay process is still running: schedule `ProcessRunner.stop()` on the Tk thread (non-blocking; process exit will eventually fire `on_exit_tk` again, but with `_active` already handled — see idempotence below).
  - If `match_started` is True: this is a real match end (or mid-match crash). Post match-ended message (with duration on success, scrubbed tail on failure), clear slot, clear log watcher, set ready presence. Done.
  - If `match_started` is False AND `attempt < max_attempts`: increment `attempt`, `await asyncio.sleep(2)`, call `_start_attempt(first=False)`. Slot stays claimed.
  - If `match_started` is False AND `attempt == max_attempts`: post final failure, clear slot, clear log watcher, set ready presence. Done.
  - **Idempotence:** if `_active` is already None when this fires (e.g. timeout killed the process, and then the process's real exit fires this again), no-op immediately.
- **`_on_spawn_failed`** — unchanged. Spawn failure is a hard error (bad path, bad args) — retrying won't help.

### `DiscordTab` — one added field

- `max_attempts_var` — a `tk.StringVar` bound to a `ttk.Spinbox(from_=1, to=5, ...)`.
- `_values()` returns `'max_attempts': self.max_attempts_var.get()` (as string; parsed to int at request time).
- `_validate()` treats blank as "1", non-integer as error.
- `_on_start()` reads it, passes as new arg to `DiscordBotThread.start(...)`.

### `DiscordBotThread.start` — signature extension

Add `max_attempts: int` parameter. Store on `self._max_attempts`. Used when constructing new `MatchRequest`s.

## Discord message catalog (final)

Every message the bot sends is now:

| Trigger | Message |
|---|---|
| Request accepted (attempt 1) | `**@user** vs \`code\` as \`char\` — starting…\nEnter my code in your Slippi game (Direct Connect): **\`BOT#123\`**` |
| Retrying | `Opponent didn't join, retrying (N/M)…` |
| Match live (marker fires) | `**match live**` |
| Match ended, success | `Match ended after 4m 12s.` |
| Match ended, mid-match crash | `Match failed (exit code N).` + scrubbed stderr tail in a code block |
| All attempts exhausted | `Opponent didn't join after N attempts.` |
| Busy | `Bot busy — @X is playing vs \`code\`.` |
| Failed to spawn | `Failed to spawn netplay: <scrubbed err>` |
| Ephemeral / validation errors | (existing behavior, unchanged) |

Everything else — Dolphin launch, menu state, character select, per-frame anything — stays silent.

## Data flow

Same as the current bot up to the point where `_start_attempt` is first called. From there:

```
_start_attempt(first=True)
  ├─ (if not first) post retry message
  ├─ install log watcher
  ├─ schedule ProcessRunner.start via root.after
  └─ create _connect_timeout task

┌─ marker fires ────────────────► _on_match_started
│                                    ├─ set match_started=True
│                                    ├─ cancel timeout task
│                                    └─ post "match live"
│
├─ timeout fires ──────────────────► _end_attempt(reason='timeout')
│                                       ├─ if match_started: ignore (marker beat timeout)
│                                       ├─ if attempts left: kill, wait 2s, _start_attempt(first=False)
│                                       └─ else: post "opponent didn't join after N attempts", clear slot
│
└─ process exits ──────────────────► _end_attempt(reason='exit', exit_code=..., recent_lines=...)
                                        ├─ if match_started: post duration/failure, clear slot
                                        ├─ if attempts left: wait 2s, _start_attempt(first=False)
                                        └─ else: post "opponent didn't join after N attempts", clear slot
```

## Timing details

- **Duration measurement.** `MatchRequest.started_at` (already exists, set to `time.monotonic()`) is now interpreted as "time the successful attempt reached `match_started=True`", NOT "time the request was received." Set it in `_on_match_started`, not in `_handle_play`. `Match ended after Xm Ys` uses `time.monotonic() - request.started_at` at end.
- **Inter-retry pause.** `await asyncio.sleep(2)` before calling `_start_attempt(first=False)` — gives Dolphin a moment to fully die before we spawn the next instance.
- **Timeout task lifecycle.** Cancelled in `_on_match_started` (existing) AND at the top of `_end_attempt` to prevent stale timeouts from firing after retry.

## Error handling

Unchanged from the current bot for anything not on the retry path:

- Bad connect code → ephemeral reply (existing).
- Unsupported char → ephemeral reply (existing).
- Non-allowlisted channel → silent drop (existing).
- Busy → public "Bot busy" (existing).
- Token invalid → status label "bad token" (existing).

New:

- `max_attempts` value out of range or non-integer → validation error at bot-start time. Bot won't start until fixed.
- Retry-in-flight race: if requester tries a new `/mention` while the bot is between retries (i.e. `_active` is still set, no netplay process running), they see "Bot busy — @you is playing vs `code`" (their own request). That's correct behavior.

## Testing

- **Unit tests** (added to `tests/test_launcher_command.py`):
  - `MatchRequest` constructs with new fields defaulting correctly.
  - Duration formatting: `_format_duration(4*60 + 12) == '4m 12s'`, `_format_duration(45) == '0m 45s'`, `_format_duration(65*60) == '65m 0s'`.
  - Max-attempts validation: parses valid strings, rejects blank/non-integer/out-of-range.
- **No live-Discord tests** — same rationale as the parent spec. Manual smoke test:
  1. Set max attempts = 2, connect timeout = 30 s.
  2. Trigger `/mention` with a made-up connect code.
  3. Watch for retry message after 30 s.
  4. Watch for final "didn't join after 2 attempts" after 60 s.
  5. Trigger with a real code, complete a match, verify "Match ended after Xm Ys" appears.

## Files touched

- **Modify** `scripts/launcher.py`:
  - Extend `MatchRequest` (3 new fields).
  - Add `_format_duration` module-level helper.
  - Add `_start_attempt` and `_end_attempt` methods on `DiscordBotThread`.
  - Refactor `_handle_play`, `_on_match_started`, `_connect_timeout`, `on_exit_tk` closure to route through `_end_attempt`.
  - Add `max_attempts_var` and Spinbox to `DiscordTab`, wire into `_values` / `_validate` / `_on_start`.
  - Extend `DiscordBotThread.start(...)` signature with `max_attempts`.
- **Modify** `tests/test_launcher_command.py`:
  - Update `MatchRequest` construction test to cover new fields.
  - Add `DurationFormatTest`, `MaxAttemptsValidationTest`.

No new dependencies.

## Open questions

None load-bearing. The 2-second inter-retry pause is a reasonable default; if it turns out to be too short (Dolphin process hanging on shutdown), bump to 5 s in a follow-up.
