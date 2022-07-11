ray submit clusters/rllib.yaml slippi_ai/rllib/run.py \
  --dolphin.path /install/squashfs-root/usr/bin/ \
  --dolphin.iso /install/SSBM.iso \
  --dolphin.headless True \
  --ray_init --wandb.use True \
  --config.num_workers 4 --env.num_envs 2
