#!/bin/bash

echo "Downloading training datasets"

curl -# --create-dirs -L \
  http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz \
  -o data/train/training-parallel-europarl-v7.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz \
  -o data/train/training-parallel-commoncrawl.tgz \
  http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz \
  -o data/train/training-parallel-nc-v10.tgz \

curl -# --create-dirs -L \
  http://www.statmt.org/wmt15/dev-v2.tgz \
  -o data/valid/dev-v2.tgz

curl -# --create-dirs -L \
  http://www.statmt.org/wmt15/test.tgz \
  -o data/test/test.tgz
