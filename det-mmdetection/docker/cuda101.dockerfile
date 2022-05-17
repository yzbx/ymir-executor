ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

# docker build -t mmcv/mmcv:gtx1080 . -f gtx1080.dockerfile
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV LANG=C.UTF-8

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
	apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC && \
	apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 \
    libxrender-dev libxext6 libgl1-mesa-glx ffmpeg sudo openssh-server \
    libyaml-dev vim tmux tree curl wget zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -U pip && \             
	pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
	pip install cython xtcocotools jupyter onnx onnx-simplifier loguru \
	tensorboard==2.5.0 numba progress yacs pthflops imagesize pydantic pytest \
	scipy pydantic pyyaml imagesize && \
	pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html && \
	pip install openmim && mim install mmdet==2.22.0

WORKDIR /app
ADD ./det-mmdetection /app
RUN mkdir -p /app/checkpoints
COPY ./det-mmdetection/*.pth /app/checkpoints/

# make PYTHONPATH include mmdetection and executor
ENV PYTHONPATH=.

# tmi framework and your app
COPY ./sample_executor /sample_executor
RUN pip install -e /sample_executor/executor
RUN mkdir /img-man
COPY ./training/mmdetection/*-template.yaml /img-man/

# dependencies: write other dependencies here (pytorch, mxnet, tensorboard-x, etc.)

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /app/ymir_start.py