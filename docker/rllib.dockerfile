FROM ubuntu:22.04

ENV INSTALL=/install
WORKDIR $INSTALL

RUN apt update
RUN apt install -y p7zip-full rsync python3-pip micro
RUN apt install python-is-python3

# big python deps
RUN pip install -U pip
RUN pip install tensorflow

# dolphin
RUN apt install -y libasound2 libegl1 libgl1 libgdk-pixbuf-2.0-0 libpangocairo-1.0-0 libusb-1.0-0
RUN pip install gdown
RUN gdown "https://drive.google.com/uc?id=1qrXsPiRD4_-voFXxIMBk2EMQ_iF8O9Me" -O dolphin
RUN chmod +x dolphin
RUN ./dolphin --appimage-extract
RUN rm dolphin
ENV DOLPHIN_PATH=$INSTALL/squashfs-root/usr/bin/

# https://serverfault.com/questions/949991/how-to-install-tzdata-on-a-ubuntu-docker-image
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt install -y tzdata

# aws tools (to pull SSBM.iso)
RUN apt install -y awscli

# ray
ARG RAY_WHEEL="https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp310-cp310-manylinux2014_x86_64.whl"
RUN pip install -U "ray[rllib] @ $RAY_WHEEL"
# RUN pip install ray[rllib]

# slippi-ai
RUN apt install -y git
COPY requirements.txt .
RUN pip install -r requirements.txt
