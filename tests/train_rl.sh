python slippi_ai/rl/run.py \
  --config.runtime.max_step=10 \
  --config.runtime.log_interval=0 \
  --config.learner.learning_rate=0 \
  --config.teacher=slippi_ai/data/checkpoints/demo \
  --config.actor.use_fake_envs=True \
  --config.actor.num_envs=2 \
  --config.actor.rollout_length=64 \
  --config.opponent.type=self \
  --config.runtime.burnin_steps_after_reset=1 \
  --config.runtime.reset_every_n_steps=3 \
  --config.optimizer_burnin_steps=2 \
  --config.value_burnin_steps=2 \
  --config.learner.ppo.num_batches=2 \
  --wandb.mode=offline \
  "$@"
