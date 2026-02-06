# NVIDIA Ubuntu image with slippi-ai installed
# FROM nvcr.io/nvidia/cuda-dl-base:25.11-cuda13.0-devel-ubuntu24.04
# Install Python (3.12) and pip
# RUN apt update && apt install -y python3 python3-pip

# FROM nvcr.io/nvidia/jax:25.10-py3
FROM nvcr.io/nvidia/jax:26.01-py3

# Set working directory
WORKDIR /slippi-ai

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set default command
CMD ["bash"]
