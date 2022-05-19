_base_ = ['./faster_rcnn_r50_fpn_ymir_coco.py']

model = dict(
    backbone=dict(
        depth=101)
)