#!/usr/bin/env bash
#
# This script builds the jetson-inference docker container from source.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#

BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3"

echo "BASE_IMAGE=$BASE_IMAGE"
echo "TAG=jetson-inference:sem-seg"

# build the container
sudo docker build -t jetson-inference:sem-seg -f Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		.