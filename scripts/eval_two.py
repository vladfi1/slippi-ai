"""Run a game between two trained agents, or vs a human player.

To run two agents against each other:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.ai.path=/path/to/agent1 \
  --p2.ai.path=/path/to/agent2
```

To run an agent against a human player in port 1:

```shell
python scripts/eval_two.py \
  --dolphin.path=/path/to/slippi-dolphin \
  --dolphin.iso=/path/to/SSBM.iso \
  --p1.type=human \
  --p2.ai.path=/path/to/agent
```

"""

from absl import app
from absl import flags
import fancyflags as ff

from slippi_ai import dolphin as dolphin_lib
from slippi_ai import flag_utils, session

PLAYERS = {
    p: ff.DEFINE_dict(f"p{p}", **session.player_flags())
    for p in session.PORTS
}
DOLPHIN = ff.DEFINE_dict('dolphin', **session.dolphin_flags())

NUM_GAMES = flags.DEFINE_integer('num_games', None, 'Number of games to play')


def main(_):
  session.run_session(session.SessionConfig(
      players={port: player.value for port, player in PLAYERS.items()},
      dolphin=flag_utils.dataclass_from_dict(
          dolphin_lib.DolphinConfig, DOLPHIN.value),
      num_games=NUM_GAMES.value,
  ))

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None
  app.run(main)
