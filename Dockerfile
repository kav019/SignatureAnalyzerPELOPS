FROM nvidia/cuda@sha256:a6a8417cb56c9a5d30c4d8c78ad18bc9b75ffe4453fe1c04b3149b3741518b06
MAINTAINER Shankara Anand

# -----------------------------
# Install Basics
# -----------------------------
RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get update && apt-get install -y \
        apt-transport-https \
        build-essential \
        cmake \
        curl \
        libboost-all-dev \
        libbz2-dev \
        libcurl3-dev \
        liblzma-dev \
        libncurses5-dev \
        libssl-dev \
        python3 \
        python3-pip \
        sudo \
        unzip \
        wget \
        zlib1g-dev \
        ghostscript \
        pkg-config \
        libhdf5-dev
#    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
#    apt-get clean && \
#    apt-get autoremove -y && \
#    rm -rf /var/lib/{apt,dpkg,cache,log}/

RUN python3 -m pip install --upgrade setuptools

# -----------------------------
# Install Signature Analyzer
# -----------------------------
RUN mkdir signatureanalyzer
COPY . /signatureanalyzer/
ENV PIP_DEFAULT_TIMEOUT 120
ENV PYTHONPATH /signatureanalyzer
RUN python3 -m pip install -e ./signatureanalyzer/.
