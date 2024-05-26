python slippi_ai/rl/run.py \
  --config.runtime.max_step=10 \
  --config.runtime.log_interval=0 \
  --config.learner.learning_rate=0 \
  --config.runtime.use_fake_data=True \
  --config.agent.path=data/checkpoints/demo \
  --config.actor.num_envs=1 \
  --config.actor.rollout_length=64 \
  --config.runtime.burnin_steps_after_reset=1 \
  --config.optimizer_burnin_steps=2 \
  --config.value_burnin_steps=2 \
  --config.learner.ppo.num_batches=2 \
  --wandb.mode=offline \
  "$@"
