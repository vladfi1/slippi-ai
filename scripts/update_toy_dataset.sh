#!/usr/bin/env sh

ROOT=slippi_ai/data/toy_dataset

python slippi_db/parse_local.py \
  --root $ROOT --reprocess

python slippi_db/scripts/make_local_dataset.py \
  --root $ROOT --winner_only=False

rm -rf $ROOT/raw.json $ROOT/parsed.pkl $ROOT/games
mv $ROOT/Parsed $ROOT/games
