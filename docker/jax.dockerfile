# NVIDIA Ubuntu image with slippi-ai installed
# FROM nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
# Install Python (3.12) and pip
# RUN apt update && apt install -y python3 python3-pip

# FROM nvcr.io/nvidia/jax:25.10-py3
# FROM nvcr.io/nvidia/jax:26.01-py3
FROM ubuntu:24.04

RUN apt update
RUN apt install -y python3 python3-venv htop git

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir s3cmd s3fs speedtest-cli ipdb ipython

WORKDIR /root
RUN git clone https://github.com/vladfi1/slippi-ai.git --branch nash-rebase

WORKDIR /root/slippi-ai

RUN pip install --no-cache-dir -r jax-requirements.txt
RUN pip install --no-cache-dir -e .[jax,cuda13]

# Set default command
CMD ["bash"]
