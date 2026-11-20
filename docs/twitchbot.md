# Twitchbot and web frontend

The twitchbot lets people play phillip over Slippi netplay. Players use
Twitch chat commands (`!play <code>`, `!agent <name>`, ...) or a web page.
Both share the same sessions and the stream. The code is in `slippi_ai/tbot/`,
and the design is in `docs/plans/web_frontend.md`.

## Running

It runs as three processes:

- `scripts/tbot_backend.py`: the backend. It loads the models and runs the
  sessions. It takes the Dolphin, agent and session flags (`--dolphin.*`,
  `--models`, `--max_sessions`, ...).
- `scripts/twitchbot.py`: the Twitch chat bot. Needs `--token` or
  `TWITCHBOT_ACCESS_TOKEN`, and `--channel`.
- `scripts/tbot_web.py`: the web page. `--port` (default 8080) and `--host`
  set where it listens. The default host, `127.0.0.1`, accepts only local
  connections, which is what the Cloudflare tunnel below needs. `0.0.0.0`
  makes the page reachable from the local network, for testing. `--channel`
  is the Twitch channel the page links to.

Start the backend first, then either or both frontends, for example in
separate windows of a tmux session:

```
python scripts/tbot_backend.py --dolphin.path=... --models=deployed_models ...
python scripts/twitchbot.py --channel=x_pilot_bot
python scripts/tbot_web.py --channel=x_pilot_bot
```

The backend runs the session manager as a named actor in a Ray cluster, and
the frontends join that cluster to call it. A frontend can be stopped,
changed and restarted without affecting games in progress. Stopping the
backend (Ctrl-C) stops all sessions.

By default the backend starts its own Ray cluster, which stops with it. The
frontends then lose their connection and exit about a minute later, so they
have to be restarted along with the backend. To restart the backend without
restarting the frontends, start a Ray cluster yourself and run the backend in
it:

```
ray start --head --include-dashboard=false
python scripts/tbot_backend.py --ray_address=auto ...
```

While the backend is down, the frontends tell users that the bot is down, and
they find the new backend once it is up. `ray stop` stops the cluster. The
backend passes `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_FLAGS` and `TF_XLA_FLAGS`
on to the sessions; other environment variables reach them only if they were
set when the cluster was started.

## Public access with a Cloudflare tunnel

A [Cloudflare tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
serves the page at a public HTTPS address such as `https://phillip2.ssbm.ai`.
It needs no port forwarding and doesn't reveal the machine's IP address.
`cloudflared` connects out to Cloudflare and passes requests to the bot on
`127.0.0.1`, so the bot stays unreachable from the network.

You need a domain whose DNS is managed by Cloudflare. The examples use
`ssbm.ai` and a tunnel named `phillip2`.

### 1. Install cloudflared

On Ubuntu, from Cloudflare's apt repository (see Cloudflare's downloads page
for other systems):

```
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update && sudo apt-get install cloudflared
```

### 2. Log in

```
cloudflared tunnel login
```

Open the URL it prints, choose the domain and approve. This saves
`~/.cloudflared/cert.pem`, which lets `cloudflared` create tunnels and DNS
records for the domain.

### 3. Create the tunnel and its DNS record

```
cloudflared tunnel create phillip2
cloudflared tunnel route dns phillip2 phillip2.ssbm.ai
```

`create` prints the tunnel's id and writes its credentials to
`~/.cloudflared/<tunnel-id>.json`. Keep that file private: anyone who has it
can serve traffic for the hostname. `route dns` adds a record that points the
hostname at the tunnel.

### 4. Configure it

Write `~/.cloudflared/config.yml`, using the id from step 3:

```yaml
tunnel: <tunnel-id>
credentials-file: /home/<user>/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: phillip2.ssbm.ai
    service: http://127.0.0.1:8080  # must match tbot_web.py's --port
  - service: http_status:404
```

The last rule returns 404 for any other hostname. Check the config with:

```
cloudflared tunnel ingress validate
cloudflared tunnel ingress rule https://phillip2.ssbm.ai/
```

### 5. Run it

Start the tunnel alongside the bot, for example in another window of the same
tmux session:

```
cloudflared tunnel run phillip2
```

It should log four "Registered tunnel connection" lines. After that,
`https://phillip2.ssbm.ai` shows the page. A new DNS record can take a minute
to resolve. The tunnel is only useful while the bot is up, so start and stop
them together. While the bot is down, the page shows a Cloudflare error.

Starting it by hand is the default. To have it start on boot instead, run it
as a systemd service:

- System service (needs sudo). Under `sudo`, `~` is root's home directory, so
  copy the config to `/etc/cloudflared/` first:
  `sudo mkdir -p /etc/cloudflared && sudo cp ~/.cloudflared/config.yml
  /etc/cloudflared/ && sudo cloudflared service install`. The service then
  uses that copy of the config.
- User service (no sudo). Write a unit to
  `~/.config/systemd/user/cloudflared.service` whose `ExecStart` is
  `cloudflared tunnel run phillip2`, then run
  `systemctl --user enable --now cloudflared`. User services stop when you
  log out unless lingering is enabled (`loginctl enable-linger`).

### Notes

- The page's live updates use server-sent events (`/api/events`). Cloudflare
  closes connections that are idle for about 100 seconds, so the server sends
  a keepalive every 15 seconds.
- The tunnel only exposes the port in `config.yml`. Ray's ports and dashboard
  stay unreachable from the internet unless something else forwards them.
- The page has no login and no limits beyond `--max_sessions`, which it
  shares with Twitch chat. Each browser is identified by a random token that
  it keeps. Every visitor who changes a setting adds an entry to
  `user_prefs.json` in the backend's `--state_dir`. Entries for web users who
  haven't played, changed a setting or loaded the page for the backend's
  `--web_prefs_days` (30 by default) are removed. Twitch users' entries are
  kept.
- To take the page down, stop the tunnel. To remove it completely, also run
  `cloudflared tunnel delete phillip2` and delete the DNS record in the
  Cloudflare dashboard.
