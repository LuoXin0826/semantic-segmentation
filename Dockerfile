FROM arm64v8/ubuntu:18.04

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

#
# build source
#
RUN mkdir docs && \
    touch docs/CMakeLists.txt && \
    sed -i 's/nvcaffe_parser/nvparsers/g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install && \
    /bin/bash -O extglob -c "cd /jetson-inference/build; rm -rf -v !(aarch64|download-models.*)" && \
    rm -rf /var/lib/apt/lists/*

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

RUN pip install -U testresources setuptools

RUN pip install -U Cython
RUN pip install -U pillow==6.1
RUN pip install -U numpy
RUN pip install -U sklearn
RUN pip install -U scipy
RUN pip install -U matplotlib
RUN pip install -U PyWavelets
RUN pip install -U kiwisolver
RUN pip install -U imagecodecs
RUN pip install -U scikit-image
RUN pip install -U h5py
Run pip install -U jupyter
Run pip install -U piexif
Run pip install -U cffi
Run pip install -U tqdm
Run pip install -U dominate
Run pip install -U tensorboardX
Run pip install -U nose
Run pip install -U ninja

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*