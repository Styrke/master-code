#!/bin/bash
VERSION=1.4.0
IMAGE=obeyed/py3-tf-gpu
ID=$(docker build  -t ${IMAGE} . | tail -1 | sed 's/.*Successfully built \(.*\)$/\1/')

echo ${ID}
docker tag ${ID} ${IMAGE}:${VERSION}
docker tag ${ID} ${IMAGE}:latest
