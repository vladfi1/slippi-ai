"""Web frontend: serves a page from which people can play phillip 2.

Needs the backend (scripts/tbot_backend.py) to be running, and can be started
and stopped without it. See docs/twitchbot.md for serving it publicly.
"""

import asyncio

from absl import app, flags
import ray

from slippi_ai.tbot import client as client_lib
from slippi_ai.tbot import web as web_lib
from slippi_ai.tbot.client import ManagerClient

PORT = flags.DEFINE_integer('port', 8080, 'Port to serve the page on.')
HOST = flags.DEFINE_string(
    'host', '127.0.0.1',
    'Address to serve the page on; 0.0.0.0 for the local network.')
CHANNEL = flags.DEFINE_string(
    'channel', 'x_pilot',
    'Twitch channel that the page links to; empty for no link.')
RAY_ADDRESS = flags.DEFINE_string(
    'ray_address', 'auto', 'Address of the Ray cluster the backend runs in.')

def main(_):
  ray.init(address=RAY_ADDRESS.value, namespace=client_lib.NAMESPACE)
  manager = ManagerClient.connect()
  try:
    asyncio.run(run(manager))
  except KeyboardInterrupt:
    pass

async def run(manager: ManagerClient):
  frontend = web_lib.WebFrontend(manager, twitch_channel=CHANNEL.value or None)
  runner = await web_lib.start(frontend, HOST.value, PORT.value)
  try:
    await asyncio.Event().wait()  # Serves until interrupted.
  finally:
    await runner.cleanup()

if __name__ == '__main__':
  app.run(main)
