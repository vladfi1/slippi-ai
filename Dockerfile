FROM python:3.9-buster

RUN pip install ray[default]

RUN apt update
RUN apt install -y p7zip-full rsync

# RUN pip install tensorflow

# slippi-specific

WORKDIR /install
COPY requirements.txt .
RUN pip install -r requirements.txt

# build the wheel externally with `maturin build` in the peppy-py repo
# ARG PEPPI_PY_WHL=peppi_py-0.4.3-cp39-abi3-linux_x86_64.whl
# COPY docker/$PEPPI_PY_WHL .
# RUN pip install $PEPPI_PY_WHL

# use peppi-py from pypi
RUN pip install peppi-py
