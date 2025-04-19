# Slippi-AI (Phillip II)

This project is the successor to [Phillip](https://github.com/vladfi1/phillip). While the original Phillip used pure deep RL, this one starts with behavioral cloning on slippi replays, which makes it play a lot more like a human. There is a [discord channel](https://discord.gg/hfVTXGu) for discussion/feedback/support.

## Playing the Bot

The bot is available to play via netplay on my [twitch channel](https://twitch.tv/x_pilot).

I am hesitant to release any trained agents as I don't want people using them on ranked/unranked, so at the moment the bot isn't available to play against locally.

## Recordings

Phillip has played a number of top players:
* [Zain 1](https://www.youtube.com/watch?v=c8nRFAGvr2c), [Zain 2](https://www.youtube.com/watch?v=XBHaHlC3_p4)
* [Amsa + Cody](https://www.youtube.com/watch?v=WGsN7lWBQP)
* [Moky](https://www.youtube.com/watch?v=1kviVflqXc4)
* [Aklo](https://www.youtube.com/watch?v=OGOEqhMptq0)

My [youtube channel](https://www.youtube.com/channel/UCzpDWSOtWpDaNPC91dqmPQg) also has some recordings and clips.

## Acknowledgements

* Huge thanks to Fizzi for writing the fast-forward gecko code that significantly speeds up RL training, for providing most of the imitation training data in the form of anonymized ranked collections (link in the Slippi discord), and of course for giving us Slippi in the first place. Even prior to rollback netcode, slippi replays were what rekindled my interest in melee AI, and are what gave name to this repo.
* Big thanks also to [altf4](https://github.com/altf4) for creating the [libmelee](https://github.com/altf4/libmelee) interface to slippi dolphin, making melee AI development accessible to everyone.
* Thank you to the many players who have generously shared their replays.
* Finally, a big thank you to my dad for proving the computing hardware used to train phillip.

# Code Overview

Phillip is trained in two stages. In the first stage, it learns to imitate human play from a large dataset of slippi replays. The resulting imitation policy is ok, but makes a lot of mistakes. In the second stage, the imitation policy is refined by playing against itself with Reinforcement Learning. This results in much stronger agents that have their own style of play.

## Creating a Dataset

The first step is preprocess your slippi replays using [`slippi_db/parse_local.py`](https://github.com/vladfi1/slippi-ai/blob/main/slippi_db/parse_local.py). See the documentation in that file for more details.

Note: local parsing currently depends on [peppi-py](https://github.com/hohav/peppi-py) version [0.6.0](https://github.com/hohav/peppi-py/commit/8c02a4659c3302321dfbfcf2093c62f634e335f7) which you may need to build manually.

The output of this step will be a `Parsed` directory of preprocessed games and a `meta.json` metadata file.

## Imitation Learning

The entry point for imitation learning is [`scripts/train.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/train.py). See [`scripts/imitation_example.sh`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/imitation_example.sh) for appropriate arguments.

Metrics are logged to [wandb](https://wandb.ai/) during training. To use your own wandb account, set the `WANDB_API_KEY` environment variable. The key metric to look at is `eval.policy.loss` -- once this has plateaued you can stop training. On a good GPU (e.g. a 3080Ti), imitation learning should take a few days to a week. The agent checkpoint will be periodically written to `experiments/<tag>/latest.pkl`.

## Reinforcement Learning

There are two entry points for RL: [`slippi_ai/rl/run.py`](https://github.com/vladfi1/slippi-ai/blob/main/slippi_ai/rl/run.py) for training an agent in the ditto, and [`slippi_ai/rl/train_two.py`](https://github.com/vladfi1/slippi-ai/blob/main/slippi_ai/rl/train_two.py) which trains two agents simultaneously. The arguments are similar for both; see [`scripts/rl_example.sh`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/rl_example.sh) for an example ditto training script.

## Evaluation

To play a trained agent or watch two trained agents play each other, use [`scripts/eval_two.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/eval_two.py). To do a full evaluation of two agents against each other, use [`scripts/run_evaluator.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/run_evaluator.py).
