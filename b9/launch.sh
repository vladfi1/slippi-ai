#! /usr/bin/env bash

set -e -x

# docker run command

# docker run -v mycode:/workspace/usr -v mydata:/workspace/AllCompressed -v $(pwd):/host --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.12-tf2-py3 bash /host/slippi-run.sh

# mounted dirs

ROOT=$(pwd)
USR=$ROOT/usr
SLIPPI=$USR/slippi-ai
DATA_ROOT=$USR/data
DATA_DIR=$DATA_ROOT/games
META_PATH=$DATA_ROOT

# get code from github

rm -rf $SLIPPI

# export BRANCH=main
export BRANCH=slippi_db
if [ ! -d $SLIPPI ]; then
  git clone --branch $BRANCH https://github.com/vladfi1/slippi-ai $SLIPPI
else
  pushd $SLIPPI ; git switch $BRANCH ; git pull ; popd
fi


# rm -rf $DATA_ROOT ;

if [ ! -d $DATA_DIR ]; then
  mkdir -p $DATA_ROOT
  mkdir $DATA_DIR
fi

# install additional environment, if not already here

# use virtualenv from python2, lightweight
# set pip dependencies wheel cache in persistent dir
# export XDG_CACHE_HOME=$USR
VENV=venv

pushd $USR
pip install virtualenv
if [ ! -d $VENV ]; then
  virtualenv --system-site-packages --app-data ./$VENV/venv-cache $VENV
fi
source $VENV/bin/activate
pip install -r $SLIPPI/requirements.txt
pip install -e $SLIPPI
popd

# execute task

# local: 192.168.1.5

# Configuration for experiment logging

NUM_DAYS=3
RUNTIME=$(($NUM_DAYS * 24 * 60 * 60))

python -u $SLIPPI/scripts/train.py with \
  tag=dataset_v2_vf \
  data.batch_size=512 \
  network.name=res_lstm network.res_lstm.num_layers=3 network.res_lstm.hidden_size=512 \
  controller_head.name=autoregressive controller_head.autoregressive.component_depth=2 \
  data.allowed_characters=fox data.allowed_opponents=all \
  dataset.data_dir=$DATA_DIR \
  dataset.meta_path=$META_PATH \
  runtime.max_runtime=$RUNTIME \
  runtime.log_interval=300 \
  runtime.save_interval=600 \
