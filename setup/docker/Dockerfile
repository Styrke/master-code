FROM nvidia/cuda:7.5-cudnn4-runtime

MAINTAINER Elias Obeid <ekobeid@gmail.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3-dev \ 
	python3-matplotlib \ 
	build-essential \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	gfortran \
	vim \
	git \
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
