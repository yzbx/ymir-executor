FROM ymir/mmcv:cuda101

RUN mkdir -p /root/.cache/torch/hub/checkpoints
COPY ./checkpoints/mmdet/*.pth /root/.cache/torch/hub/checkpoints/

WORKDIR /app
# entry point for your app
# the whole docker image will be started with `nvidia-docker run <other options> <docker-image-name>`
# and this command will run automatically
CMD python /app/ymir_start.py