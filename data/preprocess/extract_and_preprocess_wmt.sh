#!/bin/sh

# extract train
tar -xvf data/train/training-parallel-commoncrawl.tgz -C data/train
rm -rf data/train/*annotation # removing annotation files, don't need them
tar -xvf data/train/training-parallel-nc-v10.tgz -C data/train
tar -xvf data/train/training-parallel-europarl-v7.tgz -C data/train
mv data/train/training/* data/train
rm -rf data/train/training
python3 data/preprocess/merge_train.py


# extract valid
tar -xvf data/valid/dev-v2.tgz -C data/valid
mv data/valid/dev/* data/valid
rm -rf data/valid/dev
mv data/valid/newstest2014* data/test
mv data/valid/newsdev2015* data/test
rm -rf data/valid/*.sgm
rm -rf data/valid/*.cz

# extract test
tar -xvf data/test/test.tgz -C data/test
mv data/test/test/* data/test
rm -rf data/test/test
rm -rf data/test/*-ref*

python3 data/preprocess/handle_sgm.py
rm -rf data/test/*.sgm
# make sgm - to - plaintxt
python3 data/preprocess/preprocess.py
