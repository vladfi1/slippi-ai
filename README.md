# Slippi-AI (Phillip II)

This project is the successor to [Phillip](https://github.com/vladfi1/phillip). While the original Phillip used pure deep RL, this one starts with behavioral cloning on slippi replays, which makes it play a lot more like a human.

## Playing the Bot

I occasionally have the bot available to play via netplay on my [twitch channel](https://twitch.tv/x_pilot).

I am hesitant to release any trained agents as I don't want people using them on ranked/unranked, so at the moment the bot isn't available to play against locally.

## Recordings

My [youtube channel](https://www.youtube.com/channel/UCzpDWSOtWpDaNPC91dqmPQg) has some recordings, mainly of the original phillip. There is also a [video](https://www.youtube.com/watch?v=OGOEqhMptq0) of Aklo playing the new bot.

## Code Overview

The main entry points are [`scripts/train.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/train.py) for imitation learning, [`scripts/eval_two.py`](https://github.com/vladfi1/slippi-ai/blob/main/scripts/eval_two.py) for playing a trained agent, and [`slippi_ai/rl/run.py`](https://github.com/vladfi1/slippi-ai/blob/main/slippi_ai/rl/run.py) for self-play RL. Dataset creation is handled by [`slippi_db/parse_local.py`](https://github.com/vladfi1/slippi-ai/blob/main/slippi_db/parse_local.py).
