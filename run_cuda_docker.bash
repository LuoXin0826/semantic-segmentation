container_name=$1

xhost +local:
docker container run --rm -it nvidia-segmentation:latest

