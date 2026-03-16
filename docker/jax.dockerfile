# NVIDIA Ubuntu image with slippi-ai installed
# FROM nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
# Install Python (3.12) and pip
# RUN apt update && apt install -y python3 python3-pip

# FROM nvcr.io/nvidia/jax:25.10-py3
FROM nvcr.io/nvidia/jax:26.01-py3

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir s3cmd s3fs speedtest-cli

WORKDIR /root
RUN git clone https://github.com/vladfi1/slippi-ai.git --branch jax

WORKDIR /root/slippi-ai

RUN pip install --no-cache-dir -r jax-requirements.txt
RUN pip install --no-cache-dir -e .[jax]

# Set default command
CMD ["bash"]
