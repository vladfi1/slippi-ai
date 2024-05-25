FROM slippi-emulator:headless

ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -a

RUN conda create -n slippiai python=3.10 pip && conda clean -a

ENV CONDA_DEFAULT_ENV=slippiai
ENV PATH /opt/conda/envs/$CONDA_DEFAULT_ENV/bin:$PATH

RUN echo "source activate slippiai" > /etc/skel/.bashrc

RUN /bin/bash -c "source activate slippiai && conda install -c conda-forge cudatoolkit=11.8.0 cudnn=8.9 -y"

RUN apt-get update && apt-get install -y sudo && \
    useradd -m -s /bin/bash slippiuser && \
    echo 'slippiuser ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN echo "source /opt/conda/bin/activate slippiai" >> /home/slippiuser/.bashrc

COPY . /install
RUN chown -R slippiuser:slippiuser /install

USER slippiuser

RUN cd /install && \
    sudo apt-get update && \
    sudo apt-get install -y p7zip-full rsync && \
    pip install ray[default] -r requirements.txt peppi-py && \
    pip install .

WORKDIR /mnt/slippi-ai/

CMD ["/bin/echo", "Provide command for slippi ai training environment."]
