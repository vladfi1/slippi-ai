# Plan: web frontend for the twitchbot

Status: Steps 1-5 implemented (2026-11-20). Steps 1-2 (`slippi_ai/tbot/`
and `SessionManager`, with `scripts/twitchbot.py` as the Twitch frontend)
are tested on stream. Steps 3-5 (session logs, `slippi_ai/tbot/web.py` and
`static/index.html`) are tested against fake sessions and in headless
Chromium, not yet with real Dolphin sessions.

## Goal

A web page, similar to the Windows GUI (`slippi_ai/gui/app.py` on `main`),
from which anyone can pick an agent and have it connect to them over Slippi
netplay, sharing session slots, the stream slot and the screensaver bots with
Twitch chat. The page is public with no login. Until then it is tested on the
local network.

## Architecture

`scripts/twitchbot.py`'s `Bot` mixes Twitch command parsing with state that
both frontends must share: loaded models and ratings, user prefs, the session
table, `max_sessions`, who is on stream, the screensaver bot session and its
lock, and the chores loop. That state moves into a `SessionManager` with no
twitchio import, and each frontend becomes a thin adapter:

```
SessionManager            slippi_ai/tbot/manager.py
  agents(char, opp) -> list[AgentInfo]   # name, delay, trained-vs, rating
  play(user, connect_code, agent, stages, sd_stocks) -> PlayResult
  stop(user) / reset(user) / botmatch(user, a1, a2) / set_bots(user, a1, a2)
  status() -> ManagerStatus              # sessions, stream slot, bots lock
  prefs get/set per user; chores loop (gc, screensaver)
TwitchFrontend            scripts/twitchbot.py, parses "!play ABC#123"
WebFrontend               slippi_ai/tbot/web.py, aiohttp + static page + SSE
```

- Methods return picklable dataclasses (`PlayResult(ok, on_stream, reason)`),
  not chat strings; each frontend formats its own messages.
- Users are namespaced ids: `twitch:<name>`, `web:<token>`.
- Ray calls become `await actor.method.remote()`. Today `status`,
  `_gc_sessions` and `_stop_sessions` call `ray.get` inside async handlers,
  which stalls the event loop that both frontends will share.

### Processes and Ray

Phase 1: one process. The manager is a plain object; the twitch bot, the
aiohttp server and the chores loop run on one asyncio loop. aiohttp rather
than FastAPI because twitchio already depends on it. `--notwitch` runs only
the web frontend.

Phase 2 (optional): the manager becomes a detached, named async Ray actor
that creates the session actors, and each frontend is its own process that
finds it with `ray.get_actor`. This lets a frontend be restarted or
redeployed without killing games in progress. Restarting the manager still
does. Ray is chosen only because sessions are already Ray actors; without
that, an HTTP service the Twitch bot calls would be the conventional split.

## Public with no login

Without accounts, the connect code, a browser token and the client IP are
all there is to go on.

- Identity: the page generates a random token on first visit and keeps it in
  `localStorage`, sending it with each request; the user is `web:<token>`.
  Preferences (connect code, agent, stages, stocks) are stored under it.
  Stop/reset only work for the token that started the session.
- Web and Twitch users are treated the same: they share `max_sessions` and
  the stream slot is first come, first served, as it is today. As now, a user
  has at most one session, and a connect code is in at most one session
  across both frontends. Starting a session for someone else's code is
  harmless: they get a netplay request they can ignore, and the menu timeout
  (`menu_timeout`, 3 minutes) stops it.
- No other limits up front. Per-IP limits, rate limits or a separate web cap
  get added if abuse actually shows up.
- Not on the web: the admin commands `!kick` (mods only), `!reload`, `!gc`.
- Deployment: a Cloudflare tunnel to the bot on localhost provides HTTPS
  without port forwarding (set up as in `docs/twitchbot.md`). Ray's ports
  and dashboard are not exposed.

On the local network, the same code runs with `--web_host=0.0.0.0`.

## The page

| GUI (`app.py`) | Web |
|---|---|
| Character and opponent filters, model table (Model, Reaction delay, Trained against, Status) | Same filters; table from `manager.agents()` with a Rating column; auto-* and basic-* grouped as `group_special_agents` does |
| Opponent group (human port / CPU) | Connect code, stages multi-select, "SD to N stocks" |
| Start/Stop button and status label | Play / Stop / Reset; status line, e.g. "Connecting to ABC#123 on stream; connect to XYZ#456" |
| Log pane | Per-session log over SSE, from a ring buffer of log lines in `Session` read by a `logs(since)` actor method |
| (none) | Server status: active sessions, who is on stream, the bots lock; embedded Twitch player; Botmatch and Bots forms |

One static HTML/JS file, with no build step. Status, announcements and the
user's logs use server-sent events (`/api/events`); one poller shares the
status among all open pages. Session actors take concurrent calls
(`max_concurrency`) and the status times out on sessions still loading their
agents, so the page doesn't stall behind them.

## Steps

1. Move `AgentConfig`, `Session`, `BotSession`, `ModelCache` and `UserPrefs`
   from `scripts/twitchbot.py` into `slippi_ai/tbot/`, without changing them.
2. Extract `SessionManager` from `Bot` (structured results, async Ray calls,
   namespaced users) and reduce `Bot` to `TwitchFrontend`. Check every `!`
   command behaves as before.
3. Add the log ring buffer and `logs()` to `Session`.
4. `WebFrontend`: FastAPI routes, browser token, SSE status, enabled by a
   `--web_port` absl flag in the same process. Test on the local network.
5. The static page.
6. Before going public: Cloudflare tunnel (done 2026-11-20,
   `https://phillip2.ssbm.ai`; see `docs/twitchbot.md`).
7. (Optional) Phase 2: the manager as a detached Ray actor, with separate
   Twitch and web entry points.
