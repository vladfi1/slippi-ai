#! /usr/bin/env bash

set -e -x

# docker run command

# docker run -v mycode:/workspace/usr -v mydata:/workspace/AllCompressed -v $(pwd):/host --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.12-tf2-py3 bash /host/slippi-run.sh

# mounted dirs

ROOT=$(pwd)
USR=$ROOT/usr
SLIPPI=$USR/slippi-ai
AllCompressed=$ROOT/AllCompressed

# get code from github

export BRANCH=main

rm -rf $SLIPPI
if [ -z "$(ls -A $SLIPPI)" ]; then
 git clone --branch $BRANCH https://github.com/vladfi1/slippi-ai $SLIPPI
else
 pushd $SLIPPI ; git fetch origin; git switch $BRANCH ; git pull ; popd
fi

# get data from gdrive

# rm -r $AllCompressed

if [ ! -d $AllCompressed ]; then
 pip install gdown

 ACT=AllCompressed.tar
 DATESET_URL="https://drive.google.com/u/0/uc?id=1O6Njx85-2Te7VAZP6zP51EHa1oIFmS1B"

 gdown $DATESET_URL -O $ACT
 tar xvf $ACT
 rm $ACT
fi

# install additional environment, if not already here

# use virtualenv from python2, lightweight
# set pip dependencies wheel cache in persistent dir
export XDG_CACHE_HOME=$USR
VENV=venv

pushd $USR
# rm -rf $VENV  # purge old versions of e.g. libmelee
pip install virtualenv
if [ -z "$(ls -A $VENV/bin/activate)" ]; then
 virtualenv --system-site-packages --app-data ./$VENV/venv-cache $VENV
fi
source $VENV/bin/activate
pip install -r $SLIPPI/requirements.txt
pip install -e $SLIPPI
popd


# execute task

# local: 192.168.1.5

# S3 bucket for saving parameters, can leave blank
# export S3_CREDS=

# mongodb for experiment logging
export MONGO_URI="mongodb+srv://Stephen:XX4j_%Um5.iJsFS@cluster0.xldc0dl.mongodb.net/?retryWrites=true&w=majority"
python $SLIPPI/scripts/train.py with \
 data.max_action_repeat=0 \
 data.batch_size=512 \
 data.unroll_length=64 \
 network.name=res_lstm \
 network.res_lstm.num_layers=3 network.res_lstm.hidden_size=512 \
 controller_head.name=autoregressive \
 controller_head.autoregressive.component_depth=1 \
 data.allowed_characters=fox data.allowed_opponents=all \
 dataset.data_dir=$AllCompressed \
 epoch_time=300 num_epochs=864 save_interval=3600 save_to_s3=True