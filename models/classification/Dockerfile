FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS build

ARG PYTHON_VERSION=3.7
ARG CONDA_PYTHON_VERSION=3
ARG CONDA_DIR=/opt/conda

#Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Install miniconda
ENV PATH $CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=$CONDA_DIR/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y python=$PYTHON_VERSION && \
    conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch && \
    conda install -y h5py scikit-learn matplotlib seaborn \
    pandas mkl-service cython && \
    conda clean -tipsy

# Install apex
WORKDIR /tmp/
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
 
# Installs cloudml-hypertune for hyperparameter tuning.
# It’s not needed if you don’t want to do hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code 
COPY model.py model.py
COPY features.py features.py
COPY fgenerator.py fgenerator.py
COPY task.py task.py
COPY settings.py settings.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]
