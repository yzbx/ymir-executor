import argparse
import glob
import logging
import os
import os.path as osp
import subprocess

import yaml
from ymir_exc import env, monitor

from mmdet.core.evaluation.eval_hooks import update_training_result_file


def get_user_config(config, key_words, default_value):
    v = default_value
    for key in key_words:
        if key in config:
            v = config[key]
            return v
    return v

def get_args():
    parser = argparse.ArgumentParser('ymir mmdetection training')
    parser.add_argument('--cfg', help='the config file path', default=None)

    return parser.parse_args()

model_to_url = dict(
    faster_rcnn_r50_fpn='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    faster_rcnn_r101_fpn='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth',
    yolox_tiny='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
    yolox_s='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
    yolox_l='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    yolox_x='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
    yolox_nano='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
)

def main():
    # the directory info
    env_config = env.get_current_env()

    args = get_args()
    if args.cfg is not None:
        config_file = args.cfg
        with open(config_file, 'r', encoding="utf8") as f:
            executor_config = yaml.safe_load(f)
    else:
        # training info
        executor_config = env.get_executor_config()

    # default ymir config
    gpu_id = executor_config.get("gpu_id", '0,1')
    gpu_nums = len(gpu_id.split(","))

    classes = executor_config.get("class_names", [])
    num_classes = len(classes)
    assert num_classes>0, 'not find class_names in config file!'

    pretrained_model_paths = executor_config.get('pretrained_model_paths', [])
    # user define config
    learning_rate = get_user_config(executor_config, ['learning_rate', 'lr'], 0.01)
    batch_size = get_user_config(executor_config, ['batch_size', 'batch'], 64)
    max_epochs = get_user_config(
        executor_config, ['max_epochs', 'epoch', 'epochs', 'max_epoch'], 100)

    samples_per_gpu = max(1, batch_size//gpu_nums)
    workers_per_gpu = min(4, max(1, samples_per_gpu//2))
    model = get_user_config(executor_config, ['model','model_name'], 'yolox_nano')

    pretrain = get_user_config(executor_config, ['pretrain'], False)

    supported_models = []
    if model.startswith("faster_rcnn"):
        files = glob.glob(
            osp.join('configs/faster_rcnn/faster_rcnn_*_ymir_coco.py'))
        supported_models = ['faster_rcnn_r50_fpn', 'faster_rcnn_r101_fpn']
    elif model.startswith("yolox"):
        files = glob.glob(osp.join('configs/yolox/yolox_*_8x8_300e_ymir_coco.py'))
        supported_models = ['yolox_nano', 'yolox_tiny',
                            'yolox_s', 'yolox_m', 'yolox_l', 'yolox_x']
    else:
        files = glob.glob(osp.join('configs/*/*_ymir_coco.py'))
        supported_models = [osp.basename(f) for f in files]

    assert model in supported_models, f'unknown model {model}, not in {supported_models}'

    if pretrain:
        if len(pretrained_model_paths) > 0:
            checkpoint_file = pretrained_model_paths[0]
        else:
            checkpoint_file = model_to_url[model]
    else:
        checkpoint_file = None

    # modify base config file
    base_config_file = './configs/_base_/datasets/ymir_coco.py'

    modify_dict = dict(
        classes=classes,
        num_classes=num_classes,
        max_epochs=max_epochs,
        lr=learning_rate,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        data_root=env_config.input.root_dir,
        img_prefix=env_config.input.assets_dir,
        ann_prefix=env_config.input.annotations_dir,
        train_ann_file=env_config.input.training_index_file,
        val_ann_file=env_config.input.val_index_file,
        tensorboard_dir=env_config.output.tensorboard_dir,
        work_dir=env_config.output.models_dir,
        checkpoints_path=checkpoint_file
    )

    logging.info(f'modified config is {modify_dict}')
    with open(base_config_file, 'r') as fp:
        lines = fp.readlines()

    fw = open(base_config_file, 'w')
    for line in lines:
        for key in modify_dict:
            if line.startswith((f"{key}=", f"{key} =")):
                value = modify_dict[key]
                if isinstance(value, str):
                    line = f"{key} = '{value}' \n"
                else:
                    line = f"{key} = {value} \n"
                break
        fw.write(line)
        logging.info(line.strip())
    fw.close()

    # training
    train_config_file = ''
    for f in files:
        if osp.basename(f).startswith(model):
            train_config_file = f

    monitor.write_monitor_logger(percent=0.01)
    # /in
    work_dir = env_config.output.models_dir
    if gpu_nums == 1:
        cmd = f"python tools/train.py {train_config_file} " + \
            f"--work-dir {work_dir} --gpu-id {gpu_id}"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        cmd = f"./tools/dist_train.sh {train_config_file} {gpu_nums} " + \
            f"--work-dir {work_dir}"
    logging.info(f"run command: {cmd}")

    subprocess.check_output(cmd.split())

    # eval_hooks will generate training_result_file if current map is best.
    # create a fake map = 0 if no training_result_file generate in eval_hooks
    if not osp.exists(env_config.output.training_result_file):
        update_training_result_file(0)

    monitor.write_monitor_logger(percent=1.0)
