#!/bin/bash
VERSION=1.3.0
IMAGE=obeyed/py3-tf-gpu
ID=$(docker build  -t ${IMAGE} . | tail -1 | sed 's/.*Successfully built \(.*\)$/\1/')

docker tag ${ID} ${IMAGE}:${VERSION}
docker tag ${ID} ${IMAGE}:latest
