_base_ = './yolox_s_8x8_300e_ymir_coco.py'
checkpoints_path={{_base_.checkpoints_path}}
# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320),
    init_cfg=dict(type='Pretrained', 
            checkpoint=checkpoints_path))