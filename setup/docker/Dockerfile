FROM nvidia/cuda:7.5-cudnn4-runtime

MAINTAINER Elias Obeid <ekobeid@gmail.com>

# Pick up some TF dependencies
# was unable to install scikit-image via pip3, so try python-skimage here
RUN apt-get update && apt-get install -y \
  python3-pip \ 
  python3-dev \ 
  libjpeg-dev \ 
  libjpeg8-dev \ 
  python3-matplotlib \ 
  build-essential \ 
  libblas-dev \ 
  liblapack-dev \ 
  libatlas-base-dev \ 
  gfortran \ 
  vim \ 
  git \ 
  curl \ 
  python-skimage \ 
  && apt-get clean \ 
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp34-cp34m-linux_x86_64.whl 
# unable to install from requirements file -- installing manually
RUN pip3 install numpy scipy matplotlib sklearn click nltk pandas

CMD ["/bin/bash"]
