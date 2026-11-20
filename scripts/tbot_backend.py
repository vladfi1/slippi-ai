"""Backend for the twitchbot's frontends: runs the SessionManager.

Starts a Ray cluster with the SessionManager as a named actor. The frontends
(scripts/twitchbot.py and scripts/tbot_web.py) join the cluster and call the
actor, so they can be started and stopped while games go on. Stopping the
backend stops all sessions.
"""

import logging
import os

# Allow each agent to only use as much memory as it needs.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

from absl import app, flags
import fancyflags as ff
import ray

from slippi_ai import flag_utils, eval_lib, utils
from slippi_ai import dolphin as dolphin_lib
from slippi_ai.tbot import client as client_lib
from slippi_ai.tbot import manager as manager_lib

RAY_ADDRESS = flags.DEFINE_string(
    'ray_address', None,
    'Ray cluster to run in, e.g. "auto" for one started with `ray start`;'
    ' None starts a new one, which stops with the backend. The frontends'
    ' outlive a backend restart only in a cluster that outlives it.')

# Passed on to the sessions, whose workers don't inherit this process's
# environment when the cluster was started with `ray start`.
FORWARDED_ENV_VARS = [
    'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_FLAGS', 'TF_XLA_FLAGS']

WEB_PREFS_DAYS = flags.DEFINE_float(
    'web_prefs_days', 30,
    'Forget the preferences of web users not seen for this many days.')

STREAM = flags.DEFINE_boolean('stream', True, 'Stream one of the sessions.')

BOT_SESSION_INTERVAL = flags.DEFINE_float(
    'bot_session_interval', 1,
    'minutes before starting a bot session, negative means disabled')

BOTS_LOCK_MINUTES = flags.DEFINE_float(
    'bots_lock_minutes', 15,
    'Minutes after !bots during which only the initiator or a mod can'
    ' change the bots.')

# Bot settings
_DOLPHIN_CONFIG = dolphin_lib.DolphinConfig(
    infinite_time=False,
    save_replays=True,
    replay_dir='Replays/Twitchbot',
    console_timeout=120,  # allow opponent to pause
)
# STAGES = flags.DEFINE_multi_enum(
#     'stage', ['all'], stages.keys(), 'Stages to play on.')
DOLPHIN = ff.DEFINE_dict(
    'dolphin', **flag_utils.get_flags_from_default(_DOLPHIN_CONFIG))

MODELS_PATH = flags.DEFINE_string('models', 'pickled_models', 'Path to models')
STATE_DIR = flags.DEFINE_string(
    'state_dir', '.twitchbot',
    'Directory for state that persists across restarts: the model config'
    ' cache and user preferences. Empty string disables persistence.')

EVAL_DB = flags.DEFINE_string(
    'eval_db', 'untracked/eval_db.sqlite',
    'Eval database (see scripts/eval_db.py) whose ratings pick the auto-*'
    ' agents. Falls back to the training matchups if missing or empty.')

RATING_OFFSET = flags.DEFINE_float(
    'rating_offset', 2000,
    'Added to the (mean zero) eval ratings shown by !rating and !auto.')

MIN_STREAM_RATING = flags.DEFINE_float(
    'min_stream_rating', 1800,
    'Only sessions against agents rated at least this (rating_offset'
    ' included) go on stream; unrated agents never do.')

MAX_SESSIONS = flags.DEFINE_integer('max_sessions', 4, 'Maximum number of concurrent sessions.')

# Serves as the default agent people play against, and the "screensaver" agent.
agent_flags = utils.map_nt(lambda x: x, eval_lib.AGENT_FLAGS)
agent_flags.update(
    async_inference=ff.Boolean(True),
    run_on_cpu=ff.Boolean(False),
)
agent_flags['tf']['jit_compile'] = ff.Boolean(True)
agent_flags['jax']['functionalize'] = ff.Boolean(True)

del agent_flags['path']  # not used
AGENT = ff.DEFINE_dict('agent', **agent_flags)


BOT = flags.DEFINE_string('bot', None, 'Path to first bot agent.')
BOT2 = flags.DEFINE_string('bot2', None, 'Path to second bot agent.')

DEFAULT_AGENT = flags.DEFINE_string('default_agent', None, 'Name of default agent.')

COSTUME = flags.DEFINE_integer(
    'costume', 0, 'Costume index for the bot when playing against humans.')

BOTMATCH_REPLAY_DIR = flags.DEFINE_string(
    'botmatch_replay_dir', 'Replays/BotMatch',
    'Base directory for user bot-match replays.')
BOTMATCH_MAX_GAMES = flags.DEFINE_integer(
    'botmatch_max_games', 25,
    'Max games per bot-match session before it auto-stops.')

def main(_):
  env_vars = {
      name: os.environ[name]
      for name in FORWARDED_ENV_VARS if name in os.environ}
  ray.init(
      address=RAY_ADDRESS.value,
      namespace=client_lib.NAMESPACE,
      runtime_env=dict(env_vars=env_vars),
  )

  try:
    ray.get_actor(client_lib.ACTOR_NAME)
  except ValueError:
    pass
  else:
    raise app.UsageError('Another backend is already running in this cluster.')

  remote_manager = ray.remote(manager_lib.SessionManager)
  manager = remote_manager.options(name=client_lib.ACTOR_NAME).remote(
      models_path=MODELS_PATH.value,
      eval_db_path=EVAL_DB.value,
      rating_offset=RATING_OFFSET.value,
      min_stream_rating=MIN_STREAM_RATING.value,
      max_sessions=MAX_SESSIONS.value,
      dolphin_config=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      agent_kwargs=AGENT.value,
      stream=STREAM.value,
      bot_session_interval=BOT_SESSION_INTERVAL.value,
      bots_lock_minutes=BOTS_LOCK_MINUTES.value,
      bot=BOT.value,
      bot2=BOT2.value,
      default_agent=DEFAULT_AGENT.value,
      costume=COSTUME.value,
      botmatch_replay_dir=BOTMATCH_REPLAY_DIR.value,
      botmatch_max_games=BOTMATCH_MAX_GAMES.value,
      state_dir=STATE_DIR.value,
      prefs_max_days={'web': WEB_PREFS_DAYS.value},
  )

  try:
    # Raises if the manager fails to start, e.g. to load the models.
    ray.get(manager.info.remote())
    logging.info('Backend ready; the frontends can be started.')
    # Runs until shutdown; raises if the manager dies.
    ray.get(manager.run_chores.remote())
  except KeyboardInterrupt:
    ray.get(manager.shutdown.remote())

if __name__ == '__main__':
  app.run(main)
