python slippi_ai/rl/train_two.py \
  --config.runtime.max_step=10 \
  --config.runtime.log_interval=0 \
  --config.learner.learning_rate=0 \
  --config.p1.teacher=slippi_ai/data/checkpoints/demo \
  --config.p2.teacher=slippi_ai/data/checkpoints/demo \
  --config.actor.use_fake_envs=True \
  --config.actor.num_envs=1 \
  --config.actor.rollout_length=64 \
  --config.runtime.burnin_steps_after_reset=1 \
  --config.optimizer_burnin_steps=2 \
  --config.value_burnin_steps=2 \
  --config.learner.ppo.num_batches=2 \
  --wandb.mode=offline \
  "$@"
