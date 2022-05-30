ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

# docker build -t mmcv/mmcv:gtx3090 . -f gtx3090.dockerfile
# cuda11.3 + pytorch 1.10.0
# cuda11.1 + pytorch 1.9.0
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

RUN pip install -i https://mirrors.aliyun.com/pypi/simple -U pip && \
	pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
	pip install cython xtcocotools jupyter onnx onnx-simplifier loguru \
		tensorboard==2.5.0 numba progress yacs pthflops imagesize pydantic pytest \
		scipy pydantic pyyaml imagesize && \
	pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html && \
	pip install openmim && mim install mmdet==2.22.0

WORKDIR /app
ADD ./det-mmdetection /app
RUN mkdir -p /checkpoints
COPY ./det-mmdetection/*.pth /checkpoints/
# RUN wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O /checkpoints/resnet18-f37072fd.pth && \
# 	wget https://download.pytorch.org/models/resnet34-b627a593.pth -O /checkpoints/resnet34-b627a593.pth && \
# 	wget https://download.pytorch.org/models/resnet101-cd907fc2.pth -O /checkpoints/resnet101-cd907fc2.pth && \
# 	wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth -O /checkpoints/resnet50-11ad3fa6.pth && \
# 	wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth -O /checkpoints/yolox_tiny.pth && \
# 	wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth -O /checkpoints/yolox_s.pth && \
# 	wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth -O /checkpoints/yolox_l.pth && \
# 	wget https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth -O /checkpoints/yolox_x.pth

# make PYTHONPATH include mmdetection and executor
ENV PYTHONPATH=.

# tmi framework and your app
COPY ./sample_executor /sample_executor
RUN pip install -e /sample_executor
RUN mkdir /img-man
COPY ./training/mmdetection/*-template.yaml /img-man/

# dependencies: write other dependencies here (pytorch, mxnet, tensorboard-x, etc.)

# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /app/ymir_start.py