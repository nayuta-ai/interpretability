#!/bin/sh
#
# Run the docker container.

. docker/env.sh
docker stop $CONTAINER_NAME
docker run \
  -dit \
  --gpus all \
  -v $PWD:/workspace \
  -p 8891:8891 \
  --name $CONTAINER_NAME\
  --rm \
  --shm-size=2g \
  $IMAGE_NAME
docker exec \
  -d \
  $CONTAINER_NAME sh /workspace/docker/init.sh