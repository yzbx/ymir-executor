checkpoints_path = '/workspace/checkpoints/yolox_nano.pth' 
################################################################# supported config from ymir config.yaml
# model settings
num_classes = 19 
classes = ['tvmonitor', 'sofa', 'sheep', 'pottedplant', 'person', 'motorbike', 'horse', 'dog', 'diningtable', 'cow', 'chair', 'cat', 'car', 'bus', 'bottle', 'boat', 'bird', 'bicycle', 'aeroplane'] 

# dataset settings
dataset_type = 'YMIR_Coco_Dataset'
data_root = '/in' 
img_prefix = '/in/assets' 
ann_prefix = '/in/annotations' 
train_ann_file = '/in/train-index.tsv' 
val_ann_file = '/in/val-index.tsv' 
tensorboard_dir = '/out/tensorboard' 
# optimizer
lr = 0.01 
weight_decay = 5e-4
max_epochs = 10 
samples_per_gpu = 8 
workers_per_gpu = 4 
# faster-rcnn lr=0.02, weight_decay=1e-4, max_epochs=100
# yolox lr=0.01, weight_decay=5e-4, max_epochs=300

################################################################# default schedule, modify from '../_base_/schedules/schedule_1x.py'
optimizer = dict(type='SGD',
                 lr=lr, 
                 momentum=0.9, 
                 weight_decay=weight_decay)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

################################################################# default runtime, modify from '../_base_/default_runtime.py'

num_last_epochs = 15 # evaluation and save checkpoints every 1 interval for the last 15 epoch
interval = 10 # evaluation and save checkpoints every 10 interval 
 
checkpoint_config = dict(interval=interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook', log_dir=tensorboard_dir)
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '/out/models' 

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

################################################################# evaluation and checkpoint modify from yolox

evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox',
    classwise=True)