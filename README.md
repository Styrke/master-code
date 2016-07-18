# Readme

This is the code repository that accompany the master's thesis by [Alexander Rosenberg Johansen](https://github.com/alrojo), [Elias Obeid](https://github.com/Obeyed), and [Jonas Meinertz Hansen](https://github.com/Styrke).

The goal of the project was to train recurrent neural netowrks to translate text between different languages.

## Setup

The code was made to be run in a docker container (see the [`setup/`` directory](setup/)) but should run equally well on any setup with the right dependencies installed.

Most of the code is written for python 3 with the following modules: TensorFlow (0.9), matplotlib, nltk, click, numpy.

Take a look at the [Dockerfile](/setup/docker/Dockerfile) to see the exact setup we were using.

We only recommend training these models on fast GPUs with lots of RAM.

## Data

The setup is made for training on the parallel data from the [WMT '15 dataset](http://www.statmt.org/wmt15/translation-task.html#download). In order to be able to train seamlessly on different machines, we have made shell script for atuomatically downloading and preprocessing the data.

Executing the [`data/get_data.sh`](data/get_data.sh) script will automatically download the major datasets of WMT '15 in many languages and save training sets in `data/train/` and validation sets in `data/valid`

The alphabets are included in this git repository, but new ones can be built based on other datasets by editing and running [`data/build_alphabet.py`](data/build_alphabet.py).

## Training models

Hyperparameters, model architecture, and other settings are controlled by config files in the [`configs/` directory](configs/). Configs can inherit settings from other configs, and most configs (e.g. [the test config](configs/test.py)) use [the default config (`default.py`)](configs/default.py) as the base.

Start training with a config file by giving its module name as argument when running `train.py`.

For example if we want to train with the config `configs/attn_units/s-050.py`, we should use the following command:

    python3 train.py attn_units.s-050

If the config has defined a name that is not `None`, then checkpoints and log files for TensorBoard are saved in subdirectories in `train/[name]/`.
