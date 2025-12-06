python scripts/run_evaluator.py \
  --fake_envs \
  --num_envs=2 \
  --rollout_length=180 \
  --player.ai.path=slippi_ai/data/checkpoints/rl_demo \
  --opponent.ai.path=slippi_ai/data/checkpoints/demo \
  "$@"
