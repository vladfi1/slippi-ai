FROM python:3.9-buster

RUN pip install ray[default]

RUN apt update
RUN apt install -y p7zip-full rsync

WORKDIR /install
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install peppi-py
