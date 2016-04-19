FROM nvidia/cuda:7.5-cudnn4-runtime

MAINTAINER Elias Obeid <ekobeid@gmail.com>

# Pick up some TF dependencies
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
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade http://ci.tensorflow.org/view/Nightly/job/nigntly-matrix-linux-gpu/TF_BUILD_CONTAINER_TYPE=GPU,TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-working/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-0.8.0rc0-cp34-cp34m-linux_x86_64.whl
# unable to install from requirements file -- installing manually
RUN pip3 install numpy scipy matplotlib sklearn pandas scikit-image click nltk

CMD ["/bin/bash"]
