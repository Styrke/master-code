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
  https://www.dropbox.com/s/ohq7mxe1bo7opnr/devtest2006.en?dl=1 \
  -o valid/devtest2006.en \
  https://www.dropbox.com/s/otepeu5njn8xjkq/devtest2006.fr?dl=1 \
  -o valid/devtest2006.fr \
  https://www.dropbox.com/s/6akjlul7lnuqrfp/devtest2006.da?dl=1 \
  -o valid/devtest2006.da \
  https://www.dropbox.com/s/tigjshr6qjc0a3p/test2006.en?dl=1 \
  -o valid/test2006.en \
  https://www.dropbox.com/s/qovckusy274b16l/test2006.fr?dl=1 \
  -o valid/test2006.fr \
  https://www.dropbox.com/s/xy8bmuvcwxkz48p/test2006.da?dl=1 \
  -o valid/test2006.da \
  https://www.dropbox.com/s/as5zwltt0x0vqya/test2007.en?dl=1 \
  -o valid/test2007.en \
  https://www.dropbox.com/s/0j026v4xxr9rwnl/test2007.fr?dl=1 \
  -o valid/test2007.fr \
  https://www.dropbox.com/s/cae54770cpsz1vv/test2007.da?dl=1 \
  -o valid/test2007.da \
  https://www.dropbox.com/s/tvajwilbkxexnbp/test2008.en?dl=1 \
  -o valid/test2008.en \
  https://www.dropbox.com/s/u1if8fi16w49w8s/test2008.fr?dl=1 \
  -o valid/test2008.fr \
  https://www.dropbox.com/s/orwkf7bfow6cwen/test2008.da?dl=1 \
  -o valid/test2008.da
