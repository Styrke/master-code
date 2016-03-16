#!/bin/bash
VERSION=1.0.0
IMAGE=obeyed/py3-tf-gpu
ID=$(docker build  -t ${IMAGE}  .  | tail -1 | sed 's/.*Successfully built \(.*\)$/\1/')

docker tag ${ID} ${IMAGE}:${VERSION}
docker tag -f ${ID} ${IMAGE}:latest
