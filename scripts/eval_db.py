"""Keep the deployed-agent evaluation database up to date.

Typical use, after adding or updating something in deployed_models/:

  python scripts/eval_db.py                # sync, run pending evals, rate, show
  python scripts/eval_db.py --mode=plan    # just list what would run
  python scripts/eval_db.py --mode=show    # latest leaderboard

Evaluations run in-process through slippi_ai.evaluation with the same
defaults as run_scripts/run_evaluator.sh; override them with --eval.<field>.
"""

# Child processes of the multiprocess sim env re-import this module, so keep
# all the heavy imports under the main guard.
if __name__ == '__main__':
  # https://github.com/python/cpython/issues/87115
  __spec__ = None

  import os

  # Evals may pit a TensorFlow agent against a JAX one in this process. Left to
  # their defaults, JAX preallocates most of the GPU and TF grabs the rest, and
  # whichever initializes second logs CUDA_ERROR_OUT_OF_MEMORY while backing
  # off. Let both allocate on demand instead (as the twitchbot does).
  os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
  os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

  import dataclasses
  import enum
  import logging
  import time

  from absl import app, flags
  import fancyflags as ff
  import melee

  from slippi_ai import data, flag_utils
  from slippi_ai.eval_db import agents, db, matchups, ratings, runner

  class Mode(enum.Enum):
    SYNC = 'sync'    # register new/updated/renamed agents
    PLAN = 'plan'    # list pending matchups
    RUN = 'run'      # run pending matchups
    RATE = 'rate'    # fit and store ratings
    SHOW = 'show'    # print the latest leaderboard
    FULL = 'full'    # sync, run, rate, show

  MODE = flags.DEFINE_enum_class('mode', Mode.FULL, Mode, 'What to do.')
  DB = flags.DEFINE_string(
      'db', 'untracked/eval_db.sqlite', 'Path to the sqlite database.')
  DEPLOYED_MODELS = flags.DEFINE_string(
      'deployed_models', 'deployed_models', 'Directory of deployed agents.')
  STRIPPED_MODELS = flags.DEFINE_string(
      'stripped_models', 'stripped_models',
      'Directory the deployed agents link into.')

  EVAL = ff.DEFINE_dict(
      'eval', **flag_utils.get_flags_from_default(runner.DEFAULT_CONFIG))
  STAGE = flags.DEFINE_enum_class(
      'stage', runner.DEFAULT_STAGE, melee.Stage, 'Stage to evaluate on.')
  POOL = flags.DEFINE_enum(
      'pool', 'strong', list(matchups.POOLS),
      'Which agents to evaluate, as players and opponents. "strong" leaves '
      'out imitation agents and RL agents below master (bronze, silver, gold, '
      'medium, plat, diamond); "ladder" is only the ranked multi-character '
      'agents, weak tiers included; "strong-ladder" is the ranked '
      'multi-character agents from master up; "all" is everything.')
  MAX_TIER_GAP = flags.DEFINE_integer(
      'max_tier_gap', 1,
      'Only match agents whose ladder tiers are at most this far apart '
      f'(tiers: {", ".join(matchups.TIERS)}; medium counts as plat and '
      'single-character RL agents as super-gm). Negative disables the check. '
      'Untiered agents (imitation) are never restricted.')
  ALLOWED_CHARS = flags.DEFINE_list(
      'allowed_chars', [],
      'Only play these characters (libmelee names); empty means any.')
  ALLOWED_OPPONENTS = flags.DEFINE_list(
      'allowed_opponents', [],
      'With --allowed_chars, one side plays from --allowed_chars and the '
      'other from this list, in either port order. E.g. --allowed_chars=fox '
      '--allowed_opponents=falco targets Fox vs Falco only.')
  MIN_LANES_PER_PAIR = flags.DEFINE_integer(
      'min_lanes_per_pair', 8,
      'An eval between two agents cycles its envs over their pending '
      'character pairs; cap the pairs per eval so each gets at least this '
      'many envs.')
  MAX_EVALS = flags.DEFINE_integer(
      'max_evals', 0, 'Stop after this many evals; 0 means run everything.')
  SUBPROCESS = flags.DEFINE_boolean(
      'subprocess', False,
      'Run each eval in a child process (slower startup, crash isolation).')
  EVAL_LOG_DIR = flags.DEFINE_string(
      'eval_log_dir', 'untracked/eval_db_logs',
      'Where --subprocess evals write their logs and results.')
  RESET_STALE = flags.DEFINE_boolean(
      'reset_stale', False,
      'Mark evals left "running" by a dead session as failed so they rerun. '
      'Only use when no other eval_db process is active.')

  ALPHA = flags.DEFINE_float(
      'alpha', 0.01, 'Bradley-Terry prior strength (choix alpha).')
  METHOD = flags.DEFINE_enum(
      'method', 'opt', ['opt', 'mm', 'ilsr'], 'choix fitting method.')
  INCLUDE_INACTIVE = flags.DEFINE_boolean(
      'include_inactive', False,
      'Rate agents no longer deployed too, not just active ones.')
  TOP = flags.DEFINE_integer('top', 0, 'Rows to show; 0 means all.')

  def params() -> runner.EvalParams:
    config = dataclasses.replace(runner.DEFAULT_CONFIG, **EVAL.value)
    return runner.EvalParams(config=config, stage=STAGE.value)

  def character_set(names: list[str]) -> frozenset[str] | None:
    if not names:
      return None
    return frozenset(
        data.name_to_character[name.strip().lower()].name for name in names)

  def filters() -> matchups.MatchupFilters:
    if ALLOWED_OPPONENTS.value and not ALLOWED_CHARS.value:
      raise ValueError('--allowed_opponents requires --allowed_chars.')
    return matchups.MatchupFilters(
        max_tier_gap=None if MAX_TIER_GAP.value < 0 else MAX_TIER_GAP.value,
        allowed_chars=character_set(ALLOWED_CHARS.value),
        allowed_opponents=character_set(ALLOWED_OPPONENTS.value),
    )

  def pending(conn) -> list[matchups.Matchup]:
    num_envs = params().config.num_envs
    max_pairs = max(1, num_envs // max(1, MIN_LANES_PER_PAIR.value))
    return matchups.pending_matchups(
        conn, params().key(), pool=POOL.value, filters=filters(),
        max_pairs_per_eval=max_pairs)

  def do_sync(conn):
    report = agents.sync(conn, DEPLOYED_MODELS.value, STRIPPED_MODELS.value)
    print('sync:', report.summary())
    for label, items in (
        ('added', [h[:8] for h in report.added]),
        ('updated', report.updated),
        ('renamed', report.renamed),
        ('removed', report.removed),
        ('skipped', report.skipped)):
      for item in items:
        print(f'  {label}: {item}')

  def describe(conn, matchup: matchups.Matchup) -> str:
    p1 = db.display_name(conn, matchup.p1_hash)
    p2 = db.display_name(conn, matchup.p2_hash)
    pairs = matchup.character_pairs
    if len(pairs) <= 3:
      shown = ', '.join(f'{a.lower()}-{b.lower()}' for a, b in pairs)
    else:
      shown = f'{len(pairs)} character pairs'
    return f'{p1} vs {p2} ({shown})'

  def do_plan(conn) -> list[matchups.Matchup]:
    todo = pending(conn)
    active = db.active_agents(conn)
    pool = matchups.select_pool(conn, active, POOL.value)
    tiers = matchups.agent_tiers(conn, pool)
    by_tier = {}
    for h, tier in tiers.items():
      by_tier.setdefault(tier or 'untiered', []).append(db.display_name(conn, h))
    num_pairs = sum(len(m.character_pairs) for m in todo)
    print(f'plan: {len(active)} active agents, {len(pool)} in the pool, '
          f'{num_pairs} pending player pairs in {len(todo)} evals')
    if not active:
      print(f'  ({DB.value} has no agents; run --mode=sync first)')
    for tier in [*matchups.TIERS, 'untiered']:
      if tier in by_tier:
        names = sorted(by_tier[tier])
        shown = ', '.join(names[:8]) + (' ...' if len(names) > 8 else '')
        print(f'  {tier}: {len(names)} agents ({shown})')
    limit = MAX_EVALS.value or len(todo)
    for matchup in todo[:limit]:
      print('  ' + describe(conn, matchup))
    return todo

  def do_run(conn):
    if RESET_STALE.value:
      n = runner.reset_stale_running(conn)
      if n:
        print(f'reset {n} stale running evals')
    todo = pending(conn)
    if MAX_EVALS.value:
      todo = todo[:MAX_EVALS.value]
    print(f'run: {len(todo)} evals')
    run_start = time.perf_counter()
    for i, matchup in enumerate(todo):
      start = time.perf_counter()
      if SUBPROCESS.value:
        eval_id = runner.run_eval_subprocess(
            conn, matchup, params(), STRIPPED_MODELS.value, EVAL_LOG_DIR.value)
      else:
        eval_id = runner.run_eval(
            conn, matchup, params(), STRIPPED_MODELS.value)
      row = conn.execute(
          'SELECT * FROM evals WHERE id = ?', (eval_id,)).fetchone()
      elapsed = time.perf_counter() - start
      if row['status'] == 'done':
        outcome = (
            f'{row["p1_wins"]}-{row["p2_wins"]}-{row["ties"]} '
            f'({row["num_games"]} games)')
      else:
        outcome = row['status']
      # Evals share num_envs and rollout_length, so the mean so far is a fair
      # predictor of the rest.
      remaining = len(todo) - (i + 1)
      mean = (time.perf_counter() - run_start) / (i + 1)
      eta = f', ~{format_duration(mean * remaining)} left' if remaining else ''
      print(
          f'[{i + 1}/{len(todo)}] eval {eval_id}: {describe(conn, matchup)}: '
          f'{outcome} in {elapsed:.0f}s{eta}', flush=True)

  def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
      return f'{hours}h{minutes:02d}m'
    if minutes:
      return f'{minutes}m{seconds:02d}s'
    return f'{seconds}s'

  def print_leaderboard(rows):
    if not rows:
      print('no ratings yet')
      return
    limit = TOP.value or len(rows)
    name_width = max(len(r['name']) for r in rows[:limit])
    print(f'{"rank":>4} {"rating":>8} {"games":>6} {"win%":>6}  '
          f'{"name":<{name_width}}  {"character":<12}  {"tier":<8}  type  delay')
    for rank, r in enumerate(rows[:limit], 1):
      win = f'{100 * r["win_rate"]:.1f}' if r['win_rate'] is not None else '-'
      inactive = '' if r['active'] else ' (retired)'
      print(f'{rank:>4} {r["rating"]:>8.0f} {r["num_games"]:>6} {win:>6}  '
            f'{r["name"] + inactive:<{name_width}}  '
            f'{r["character"].lower():<12}  {r["tier"] or "-":<8}  '
            f'{r["agent_type"][:4]}  {r["delay"]:>5}')

  def do_rate(conn):
    run_id, result = ratings.compute_and_store(
        conn, active_only=not INCLUDE_INACTIVE.value,
        alpha=ALPHA.value, method=METHOD.value)
    print(
        f'rate: run {run_id}: {len(result.ratings)} players, '
        f'{result.num_evals} evals, {result.num_games_total} games')
    print_leaderboard(ratings.leaderboard(conn, run_id))

  def do_show(conn):
    run_id = ratings.latest_run_id(conn)
    if run_id is None:
      print('no ratings yet')
      return
    row = conn.execute(
        'SELECT * FROM rating_runs WHERE id = ?', (run_id,)).fetchone()
    print(f'rating run {run_id} from {row["computed_at"]} '
          f'({row["num_evals"]} evals, {row["num_games"]} games)')
    print_leaderboard(ratings.leaderboard(conn, run_id))

  def main(_):
    logging.getLogger().setLevel(logging.INFO)
    conn = db.connect(DB.value)
    rekeyed = runner.rekey_evals(conn)
    if rekeyed:
      print(f'updated the coverage key of {rekeyed} evals')
    mode = MODE.value
    if mode is Mode.SYNC:
      do_sync(conn)
    elif mode is Mode.PLAN:
      do_plan(conn)
    elif mode is Mode.RUN:
      do_run(conn)
    elif mode is Mode.RATE:
      do_rate(conn)
    elif mode is Mode.SHOW:
      do_show(conn)
    elif mode is Mode.FULL:
      do_sync(conn)
      do_run(conn)
      do_rate(conn)
    conn.close()

  app.run(main)
