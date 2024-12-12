#!/usr/bin/env sh

# These parameters are optimized for an i7-11700K and RTX 3080Ti with 64GB of RAM.
# I suggest increasing the num_envs until you run out of RAM, and then increasing
# the rollout_length until you run out of GPU memory. The inner_batch_size should
# be set so that num_envs / inner_batch_size is approximately the number of CPU
# threads you have available. The rest of the parameters can be left as is.

CHAR=fox
# What player(s) from the dataset should we condition on?
# This can be a comma-separated list.
NAME="Master Player"
D=18
TAG=${CHAR}_delay_${D}

python slippi_ai/rl/run.py \
  --config.runtime.tag=$TAG \
  --config.runtime.max_step=10000 \
  --config.runtime.log_interval=300 \
  --config.dolphin.path=$DOLPHIN_PATH \
  --config.dolphin.iso=$ISO_PATH \
  --config.dolphin.console_timeout=60 \
  --config.learner.learning_rate=3e-5 \
  --config.learner.value_cost=1 \
  --config.learner.reward_halflife=4 \
  --config.learner.reward.damage_ratio=0.01 \
  --config.learner.reward.ledge_grab_penalty=0.02 \
  --config.learner.policy_gradient_weight=5 \
  --config.learner.kl_teacher_weight=3e-3 \
  --config.learner.ppo.num_epochs=2 \
  --config.learner.ppo.num_batches=16 \
  --config.learner.ppo.beta=3e-1 \
  --config.learner.ppo.epsilon=1e-2 \
  --config.learner.ppo.minibatched=False \
  --config.teacher=pickled_models/${CHAR}_d${D}_imitation \
  --config.opponent.type=self \
  --config.opponent.train=True \
  --config.actor.rollout_length=240 \
  --config.actor.num_envs=96 \
  --config.actor.inner_batch_size=10 \
  --config.actor.async_envs=True \
  --config.actor.num_env_steps=4 \
  --config.actor.gpu_inference=True \
  --config.agent.name="$NAME" \
  --config.agent.batch_steps=4 \
  --config.runtime.reset_every_n_steps=512 \
  --config.runtime.burnin_steps_after_reset=5 \
  --config.optimizer_burnin_steps=128 \
  --config.value_burnin_steps=128 \
  --wandb.name=$TAG \
  --wandb.mode=online \
  --wandb.tags=ppo \
  "$@"
