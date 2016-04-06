#!/bin/bash

curl -# --create-dirs -L \
  https://www.dropbox.com/s/ncc3ekjwts1hw8p/europarl-v7.fr-en.en?dl=1 \
  -o train/europarl-v7.fr-en.en \
  https://www.dropbox.com/s/wgoqocs3jdgphzv/europarl-v7.fr-en.fr?dl=1 \
  -o train/europarl-v7.fr-en.fr \
  https://www.dropbox.com/s/17rqqouwxoa1l98/europarl-v7.da-en.en?dl=1 \
  -o ./train/europarl-v7.da-en.en \
  https://www.dropbox.com/s/rltf6dipngwlw6e/europarl-v7.da-en.da?dl=1 \
  -o ./train/europarl-v7.da-en.da \
