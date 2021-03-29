ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

WORKDIR /mnt/jetson-sdcard/

#
# install pre-requisite packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            cmake \
            python3-opencv \
    && rm -rf /var/lib/apt/lists/*
    
# pip dependencies for pytorch-ssd
RUN pip3 install --verbose --upgrade Cython && \
    pip3 install --verbose boto3 pandas

# alias python3 -> python
RUN rm /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN apt-get update -y \
  && apt-get install -y --no-install-recommends apt-utils \
  && apt-get install -y \
    python3-dev libpython3-dev python-pil python3-tk python-imaging-tk \
    build-essential wget locales libfreetype6-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev gfortran \
    libjpeg-dev libpng-dev libopenjp2-7-dev libopenjp2-tools \
    libaec-dev libblosc-dev libbrotli-dev libbz2-dev libgif-dev \
    imagemagick liblcms2-dev libjxr-dev liblz4-dev libsnappy-dev \
    libopenjp2-7-dev libopenjp2-tools libfreetype6-dev libzstd-dev \
    libwebp-dev cmake wget bzip2 autoconf automake libtool \
    liblzma-dev libzopfli-dev

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# Building: libtiff5-dev
RUN cd /mnt/jetson-sdcard \
  && wget https://gitlab.com/libtiff/libtiff/-/archive/v4.1.0/libtiff-v4.1.0.tar.bz2 \
  && tar xvfj libtiff-v4.1.0.tar.bz2 \
  && cd libtiff-v4.1.0 \
  && ./autogen.sh \
  && ./configure \
  && make install

RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py \
  && python3 /tmp/get-pip.py \
  && pip3 install -U pip

# Building: apex
RUN cd /mnt/jetson-sdcard \
  && git clone https://github.com/NVIDIA/apex \
  && cd apex \
  && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip3 install git+https://github.com/scikit-learn/scikit-learn.git

RUN pip3 install -U testresources setuptools

RUN pip3 install -U pillow==6.1
RUN pip3 install -U numpy
RUN pip3 install -U sklearn
RUN pip3 install -U scipy
Run pip3 install -U jupyter
Run pip3 install -U piexif
Run pip3 install -U cffi
Run pip3 install -U tqdm
Run pip3 install -U dominate
Run pip3 install -U tensorboardX
Run pip3 install -U nose
Run pip3 install -U ninja

RUN apt-get update
RUN apt-get -y install python-skimage

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*