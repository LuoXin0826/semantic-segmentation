sudo docker run -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace nvidia-segmentation:latest bash

sudo docker run --runtime=nvidia --ipc=host -it --rm -p 8888:8888 -v ~/workspace/semantic-segmentation:/home/workspace -v /media/youngji/StorageDevice/data/nvidia-segmentation/data_semantics:/home/dataset  nvidia-segmentation:latest bash


