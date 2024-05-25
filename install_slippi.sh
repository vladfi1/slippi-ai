#!/bin/bash

cd /install

apt update && \
apt install -y p7zip-full rsync

pip install ray[default] -r requirements.txt peppi-py
pip install .