#!/bin/bash

echo "Downloading training datasets"
curl -# --create-dirs -L \
  https://www.dropbox.com/s/ncc3ekjwts1hw8p/europarl-v7.fr-en.en?dl=1 \
  -o data/train/europarl-v7.fr-en.en \
  https://www.dropbox.com/s/wgoqocs3jdgphzv/europarl-v7.fr-en.fr?dl=1 \
  -o data/train/europarl-v7.fr-en.fr \
  https://www.dropbox.com/s/17rqqouwxoa1l98/europarl-v7.da-en.en?dl=1 \
  -o data/train/europarl-v7.da-en.en \
  https://www.dropbox.com/s/rltf6dipngwlw6e/europarl-v7.da-en.da?dl=1 \
  -o data/train/europarl-v7.da-en.da \
  https://www.dropbox.com/s/ir3f9fnj53lj8dc/europarl-v7.de-en.de?dl=1 \
  -o data/train/europarl-v7.de-en.de \
  https://www.dropbox.com/s/e86770xbsah4bbw/europarl-v7.de-en.en?dl=1 \
  -o data/train/europarl-v7.de-en.en \
  https://www.dropbox.com/s/lmequ0njypuvp8x/news-commentary-v10.de-en.de?dl=1 \
  -o data/train/news-commentary-v10.de-en.de \
  https://www.dropbox.com/s/7ej3t9qvkbx4aj4/news-commentary-v10.de-en.en?dl=1 \
  -o data/train/news-commentary-v10.de-en.en \
  https://www.dropbox.com/s/seirqrz23ly23bp/commoncrawl.de-en.de?dl=1 \
  -o data/train/commoncrawl.de-en.de \
  https://www.dropbox.com/s/xkl6mtpoqal8g41/commoncrawl.de-en.en?dl=1 \
  -o data/train/commoncrawl.de-en.en


echo "Downloading validation datasets"
curl -# --create-dirs -L \
  https://www.dropbox.com/s/ohq7mxe1bo7opnr/devtest2006.en?dl=1 \
  -o data/valid/devtest2006.en \
  https://www.dropbox.com/s/otepeu5njn8xjkq/devtest2006.fr?dl=1 \
  -o data/valid/devtest2006.fr \
  https://www.dropbox.com/s/6akjlul7lnuqrfp/devtest2006.da?dl=1 \
  -o data/valid/devtest2006.da \
  https://www.dropbox.com/s/sc2dap3w743c83c/devtest2006.de?dl=1 \
  -o data/valid/devtest2006.de \
  https://www.dropbox.com/s/tigjshr6qjc0a3p/test2006.en?dl=1 \
  -o data/valid/test2006.en \
  https://www.dropbox.com/s/qovckusy274b16l/test2006.fr?dl=1 \
  -o data/valid/test2006.fr \
  https://www.dropbox.com/s/xy8bmuvcwxkz48p/test2006.da?dl=1 \
  -o data/valid/test2006.da \
  https://www.dropbox.com/s/ztzan17ivv7co6e/test2006.de?dl=1 \
  -o data/valid/test2006.de \
  https://www.dropbox.com/s/as5zwltt0x0vqya/test2007.en?dl=1 \
  -o data/valid/test2007.en \
  https://www.dropbox.com/s/0j026v4xxr9rwnl/test2007.fr?dl=1 \
  -o data/valid/test2007.fr \
  https://www.dropbox.com/s/cae54770cpsz1vv/test2007.da?dl=1 \
  -o data/valid/test2007.da \
  https://www.dropbox.com/s/e8shmip0ac6qupg/test2007.de?dl=1 \
  -o data/valid/test2007.de \
  https://www.dropbox.com/s/tvajwilbkxexnbp/test2008.en?dl=1 \
  -o data/valid/test2008.en \
  https://www.dropbox.com/s/u1if8fi16w49w8s/test2008.fr?dl=1 \
  -o data/valid/test2008.fr \
  https://www.dropbox.com/s/orwkf7bfow6cwen/test2008.da?dl=1 \
  -o data/valid/test2008.da \
  https://www.dropbox.com/s/qxt63bm0ch4pyk6/test2008.de?dl=1 \
  -o data/valid/test2008.de \
  https://www.dropbox.com/s/56ykwr2uwc7t0o2/newstest2013.de?dl=1 \
  -o data/valid/newstest2013.de \
  https://www.dropbox.com/s/masy9ooynt0j5np/newstest2013.en?dl=1 \
  -o data/valid/newstest2013.en \
  https://www.dropbox.com/s/p6egbpq33bbjdhd/newstest2014.deen.de?dl=1 \
  -o data/valid/newstest2014.deen.de \
  https://www.dropbox.com/s/zu79foxevdilwfi/newstest2014.deen.en?dl=1 \
  -o data/valid/newstest2014.deen.en \
  https://www.dropbox.com/s/5kei79dibyibrfa/newstest2015.deen.de?dl=1 \
  -o data/valid/newstest2015.deen.de \
  https://www.dropbox.com/s/u223u3v1b297ne8/newstest2015.deen.en?dl=1 \
  -o data/valid/newstest2015.deen.en
