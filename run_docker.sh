#!/bin/bash

set -a
set -e
# source .env

image_name=europe-west1-docker.pkg.dev/algorithmic-quartet/docker-images/retina-trainer
docker build -t $image_name:$1 -t $image_name:latest .
# docker run -e CLOUD_BUCKET -it $image_name
