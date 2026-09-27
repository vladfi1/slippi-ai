"""Evaluate two agents against each other; see slippi_ai/evaluation.py.

Flags are kept flat (--num_envs etc.) so run_scripts/run_evaluator.sh and
older invocations keep working.
"""

# Make sure not to import things unless we're the main module.
# This allows child processes to avoid importing tensorflow,
# which uses a lot of memory.

if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  import json

  from absl import app, flags
  import fancyflags as ff

  from slippi_ai import data, dolphin, evaluation, flag_utils

  default_dolphin_config = dolphin.DolphinConfig(
      infinite_time=False,
      headless=True,
  )
  DOLPHIN = ff.DEFINE_dict(
      'dolphin', **flag_utils.get_flags_from_default(default_dolphin_config))

  ROLLOUT_LENGTH = flags.DEFINE_integer(
      'rollout_length', 60 * 60, 'number of steps per rollout')
  CHUNK_LENGTH = flags.DEFINE_integer(
      'chunk_length', 0,
      'Number of steps per rollout chunk; 0 means one chunk of '
      '--rollout_length. Must divide --rollout_length. Trajectories are only '
      'used for reward sums here, so smaller chunks cut peak memory: the '
      'per-step agent outputs (notably logits) and the env state buffer are '
      'sized by chunk length rather than by the full rollout.')
  NUM_ENVS = flags.DEFINE_integer('num_envs', 1, 'Number of environments.')

  FAKE_ENVS = flags.DEFINE_boolean('fake_envs', False, 'Use fake environments.')
  SIM_ENVS = flags.DEFINE_boolean('sim_envs', False, 'Use melee-sim-light environments.')
  ASYNC_ENVS = flags.DEFINE_boolean('async_envs', False, 'Use async environments.')
  NUM_ENV_STEPS = flags.DEFINE_integer(
      'num_env_steps', 0, 'Number of environment steps to batch.')
  INNER_BATCH_SIZE = flags.DEFINE_integer(
      'inner_batch_size', -1,
      'Number of environments to run sequentially per worker; -1 means '
      'num_envs / cpu_count (one worker per CPU), or 1 when num_envs < cpu_count.')
  SWAP_PORTS = flags.DEFINE_boolean('swap_ports', True, 'Swap half of env ports.')

  USE_GPU = flags.DEFINE_boolean('use_gpu', True, 'Use GPU for inference.')
  NUM_AGENT_STEPS = flags.DEFINE_integer(
      'num_agent_steps', 0,
      'Default number of agent steps to batch; a per-agent '
      '--{player,opponent}.ai.batch_steps takes precedence.')

  player_flags = evaluation.player_flags()
  PLAYER = ff.DEFINE_dict('player', **player_flags)

  SELF_PLAY = flags.DEFINE_boolean('self_play', False, 'Self play.')
  OPPONENT = ff.DEFINE_dict('opponent', **player_flags)

  # Multi-character agents can cover several matchups in one run: envs cycle
  # through the product of the two lists. Empty means the single
  # --{player,opponent}.character.
  PLAYER_CHARACTERS = flags.DEFINE_list(
      'player_characters', [],
      'Characters for the player to cycle across envs (libmelee names).')
  OPPONENT_CHARACTERS = flags.DEFINE_list(
      'opponent_characters', [],
      'Characters for the opponent to cycle across envs (libmelee names).')

  TF_PROFILE = flags.DEFINE_boolean('tf_profile', False, 'Enable TF profiler.')
  JAX_PROFILER_DIR = flags.DEFINE_string('jax_profiler_dir', None, 'Directory for JAX profiler traces.')
  NUM_GAMES = flags.DEFINE_integer(
      'num_games', 0, 'Stop after this many initially active sim games finish.')

  QUIET = flags.DEFINE_boolean('quiet', False, 'Whether to suppress non-timing prints.')
  BURNIN = flags.DEFINE_boolean('burnin', False, 'Do a burnin unroll for better timings.')
  RESULTS_PATH = flags.DEFINE_string(
      'results_path', None,
      'Write run parameters, reward stats and completed games as JSON here.')

  def parse_characters(names: list[str]) -> list:
    return [data.name_to_character[name.lower()] for name in names]

  def main(_):
    config = evaluation.EvaluationConfig(
        num_envs=NUM_ENVS.value,
        rollout_length=ROLLOUT_LENGTH.value,
        chunk_length=CHUNK_LENGTH.value,
        num_env_steps=NUM_ENV_STEPS.value,
        num_agent_steps=NUM_AGENT_STEPS.value,
        inner_batch_size=INNER_BATCH_SIZE.value,
        swap_ports=SWAP_PORTS.value,
        fake_envs=FAKE_ENVS.value,
        sim_envs=SIM_ENVS.value,
        async_envs=ASYNC_ENVS.value,
        use_gpu=USE_GPU.value,
        self_play=SELF_PLAY.value,
        num_games=NUM_GAMES.value,
        burnin=BURNIN.value,
        quiet=QUIET.value,
        tf_profile=TF_PROFILE.value,
        jax_profiler_dir=JAX_PROFILER_DIR.value,
    )
    result = evaluation.evaluate(
        config=config,
        player_kwargs={1: PLAYER.value, 2: OPPONENT.value},
        dolphin_kwargs=dolphin.DolphinConfig.kwargs_from_flags(DOLPHIN.value),
        character_pairs=evaluation.character_pairs_from_lists({
            1: parse_characters(PLAYER_CHARACTERS.value),
            2: parse_characters(OPPONENT_CHARACTERS.value),
        }),
    )
    evaluation.print_result(result)

    if RESULTS_PATH.value:
      results = result.to_json_dict(extra_params=dict(
          stage=DOLPHIN.value['stage'],
          player=PLAYER.value,
          opponent=OPPONENT.value,
      ))
      with open(RESULTS_PATH.value, 'w') as f:
        json.dump(results, f, default=evaluation.json_default)
      print(f'Wrote results to {RESULTS_PATH.value}')

  app.run(main)
