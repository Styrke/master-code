#!/bin/bash

echo "Downloading training datasets"
curl -# --create-dirs -L \
  https://www.dropbox.com/s/ncc3ekjwts1hw8p/europarl-v7.fr-en.en?dl=1 \
  -o train/europarl-v7.fr-en.en \
  https://www.dropbox.com/s/wgoqocs3jdgphzv/europarl-v7.fr-en.fr?dl=1 \
  -o train/europarl-v7.fr-en.fr \
  https://www.dropbox.com/s/17rqqouwxoa1l98/europarl-v7.da-en.en?dl=1 \
  -o ./train/europarl-v7.da-en.en \
  https://www.dropbox.com/s/rltf6dipngwlw6e/europarl-v7.da-en.da?dl=1 \
  -o ./train/europarl-v7.da-en.da \

echo "Downloading validation datasets"
curl -# --create-dirs -L \
  https://www.dropbox.com/s/lcaxi0yvv4p6559/devtest2006.en?dl=1 \
  -o valid/devtest2006.en \
  https://www.dropbox.com/s/viqzsob7wxhjvl4/devtest2006.fr?dl=1 \
  -o valid/devtest2006.fr \
  https://www.dropbox.com/s/tbfcgig3z1a93ra/devtest2006.da?dl=1 \
  -o valid/devtest2006.da \
  https://www.dropbox.com/s/tkvelrqrmclf7d7/test2006.en?dl=1 \
  -o valid/test2006.en \
  https://www.dropbox.com/s/82czeqoxk8aladk/test2006.fr?dl=1 \
  -o valid/test2006.fr \
  https://www.dropbox.com/s/atfj1gkw9cpzs8d/test2006.da?dl=1 \
  -o valid/test2006.da \
  https://www.dropbox.com/s/8z3binjwtgqu0d1/test2007.en?dl=1 \
  -o valid/test2007.en \
  https://www.dropbox.com/s/0ynrt4dljxdr2a5/test2007.fr?dl=1 \
  -o valid/test2007.fr \
  https://www.dropbox.com/s/z3911cmw01o3tgc/test2007.da?dl=1 \
  -o valid/test2007.da \
  https://www.dropbox.com/s/9yaujye5wfxlwqy/test2008.fr?dl=1 \
  -o valid/test2008.fr \
  https://www.dropbox.com/s/uqtoxeetlon2mvo/test2008.en?dl=1 \
  -o valid/test2008.en \
  https://www.dropbox.com/s/lya9hhhfcujht4n/test2008.da?dl=1 \
  -o valid/test2008.da
