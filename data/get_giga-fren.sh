#!/bin/bash

echo "Downloading training datasets"
curl -# --create-dirs -L \
  https://www.dropbox.com/s/wzz0t30v48btqua/giga-fren.release2.en?dl=1 \
  -o train/giga-fren.en \
  https://www.dropbox.com/s/tec7cx723gcosdx/giga-fren.release2.fr?dl=1 \
  -o train/giga-fren.fr \
