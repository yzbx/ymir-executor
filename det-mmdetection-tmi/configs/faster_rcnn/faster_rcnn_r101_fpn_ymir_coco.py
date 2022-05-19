_base_ = ['./faster_rcnn_r50_fpn_ymir_coco.py']
checkpoints_path={{_base_.checkpoints_path}}
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint=checkpoints_path)))