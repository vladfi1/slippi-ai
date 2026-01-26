# NVIDIA Ubuntu image with slippi-ai installed
# For GPU-accelerated TensorFlow training
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /slippi-ai

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# The nvcr image has tensorflow 0.17, so we need a compatible version of tensorflow_probability
RUN pip install --no-cache-dir tensorflow_probability[tf]==0.24
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set default command
CMD ["bash"]
