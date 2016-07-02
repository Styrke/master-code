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
  -o data/train/commoncrawl.de-en.en \
  https://www.dropbox.com/s/g95xkb9ylla0177/commoncrawl.de-en.de.norm?dl=1 \
  -o data/train/commoncrawl.de-en.de.norm \
  https://www.dropbox.com/s/febwjhyk12raz82/commoncrawl.de-en.de.tok?dl=1 \
  -o data/train/commoncrawl.de-en.de.tok \
  https://www.dropbox.com/s/2iowmpazyipb3o6/commoncrawl.de-en.en.norm?dl=1 \
  -o data/train/commoncrawl.de-en.en.norm \
  https://www.dropbox.com/s/nuxwdpql5kj2d6w/commoncrawl.de-en.en.tok?dl=1 \
  -o data/train/commoncrawl.de-en.en.tok \
  https://www.dropbox.com/s/sw4q90slqurxc11/europarl-v7.de-en.de.norm?dl=1 \
  -o data/train/europarl-v7.de-en.de.norm \
  https://www.dropbox.com/s/mo401h5qengg432/europarl-v7.de-en.de.tok?dl=1 \
  -o data/train/europarl-v7.de-en.de.tok \
  https://www.dropbox.com/s/sqjt2cpqyngym4u/europarl-v7.de-en.en.norm?dl=1 \
  -o data/train/europarl-v7.de-en.en.norm \
  https://www.dropbox.com/s/cgcuapfvpdluxu5/europarl-v7.de-en.en.tok?dl=1 \
  -o data/train/europarl-v7.de-en.en.tok \
  https://www.dropbox.com/s/6ea5zxq8mgxqkqm/news-commentary-v10.de-en.de.norm?dl=1 \
  -o data/train/news-commentary-v10.de-en.de.norm \
  https://www.dropbox.com/s/1626gk6r90v2eiy/news-commentary-v10.de-en.de.tok?dl=1 \
  -o data/train/news-commentary-v10.de-en.de.tok \
  https://www.dropbox.com/s/le2ecd0zbichsau/news-commentary-v10.de-en.en.norm?dl=1 \
  -o data/train/news-commentary-v10.de-en.en.norm \
  https://www.dropbox.com/s/6h0eak6m6n4e0fm/news-commentary-v10.de-en.en.tok?dl=1 \
  -o data/train/news-commentary-v10.de-en.en.tok


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
  -o data/valid/newstest2015.deen.en \
  https://www.dropbox.com/s/jq3mc4gelawx3f7/newstest2013.de.norm?dl=1 \
  -o data/valid/newstest2013.de.norm \
  https://www.dropbox.com/s/vi03gbaljt142b8/newstest2013.de.tok?dl=1 \
  -o data/valid/newstest2013.de.tok \
  https://www.dropbox.com/s/4ssj8okmx32qe8j/newstest2013.en.norm?dl=1 \
  -o data/valid/newstest2013.en.norm \
  https://www.dropbox.com/s/9p3kfiyuwtocv1c/newstest2013.en.tok?dl=1 \
  -o data/valid/newstest2013.en.tok \
  https://www.dropbox.com/s/gho1ji11gzx6g9v/newstest2014.deen.de.norm?dl=1 \
  -o data/valid/newstest2014.deen.de.norm \
  https://www.dropbox.com/s/t08bihlq4456x3q/newstest2014.deen.de.tok?dl=1 \
  -o data/valid/newstest2014.deen.de.tok \
  https://www.dropbox.com/s/6eipk7ub0d9uzdf/newstest2014.deen.en.norm?dl=1 \
  -o data/valid/newstest2014.deen.en.norm \
  https://www.dropbox.com/s/wocjyezgis93od9/newstest2014.deen.en.tok?dl=1 \
  -o data/valid/newstest2014.deen.en.tok \
  https://www.dropbox.com/s/nrq35fdbknxlmlz/newstest2015.deen.de.norm?dl=1 \
  -o data/valid/newstest2015.deen.de.norm \
  https://www.dropbox.com/s/w3vt7u3mw5db6zf/newstest2015.deen.de.tok?dl=1 \
  -o data/valid/newstest2015.deen.de.tok \
  https://www.dropbox.com/s/in789z29frjq9f8/newstest2015.deen.en.norm?dl=1 \
  -o data/valid/newstest2015.deen.en.norm \
  https://www.dropbox.com/s/s898aul6obvgw4r/newstest2015.deen.en.tok?dl=1 \
  -o data/valid/newstest2015.deen.en.tok
