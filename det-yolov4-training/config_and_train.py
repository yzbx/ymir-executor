from itertools import count
import logging
import yaml
import os
import time

import numpy as np
from mxnet import nd

import train_watcher


def get_model_seen(model_path):
    with open(model_path, 'rb') as mf:
        header = nd.array(np.fromfile(mf, dtype=np.int32, count=5))
        seen_images_array = header[3]
        return int(seen_images_array.asscalar())


def get_pretrained_best_map():
    try:
        with open('/out/models/result.yaml', 'r') as f:
            result_obj = yaml.safe_load(f)
            return float(result_obj['map'])
    except (FileNotFoundError, KeyError, yaml.YAMLError):
        logging.exception(msg='error occured when get pretrained best mAP, will use default value -1')
        return -1


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


config_file = "/in/config.yaml"

with open(config_file, 'r', encoding="utf8") as f:
    config = yaml.safe_load(f)

class_names = config["class_names"]
classnum = len(class_names) 
gpus = config["gpu_id"]
anchors = config["anchors"]
task_id = config["task_id"]
image_height = config["image_height"]
image_width = config["image_width"]
learning_rate = config["learning_rate"]
max_batches = config["max_batches"]
pretrained_model_params_conf = config.get("pretrained_model_params", None)
batch = int(config["batch"])
if image_width != image_height:
    raise ValueError(f"width: {image_width} != height: {image_height}")
if batch <= 0:
    raise ValueError('invalid batch size')
subdivisions = config["subdivisions"]
warmup_iterations = config["warmup_iterations"]

# copy cfg file to /out/models/
os.system("cp ./cfg/yolov4.cfg /out/models")
os.system("cp ./cfg/coco.data /out/coco.data")

# write class names to file
if class_names is not None:
    with open("/out/coco.names", 'w') as f:
        for each_name in class_names:
            f.write(f"{each_name}\n")

if classnum != 1:
    num_filter = (5 + classnum) * 3
    os.system("sed -i 's/classes=1/classes={}/g' /out/models/yolov4.cfg".format(classnum))
    os.system("sed -i 's/filters=18/filters={}/g' /out/models/yolov4.cfg".format(num_filter))

if anchors is not None:
    os.system("sed -i 's/12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401/{}/g' /out/models/yolov4.cfg".format(anchors))

if image_height is not None:
    os.system("sed -i 's/height=608/height={}/g' /out/models/yolov4.cfg".format(image_height))

if image_width is not None:
    os.system("sed -i 's/width=608/width={}/g' /out/models/yolov4.cfg".format(image_width))

if batch is not None:
    os.system("sed -i 's/batch=64/batch={}/g' /out/models/yolov4.cfg".format(batch))

if subdivisions is not None:
    os.system("sed -i 's/subdivisions=32/subdivisions={}/g' /out/models/yolov4.cfg".format(subdivisions))

if learning_rate is not None:
    os.system("sed -i 's/learning_rate=0.0013/learning_rate={}/g' /out/models/yolov4.cfg".format(learning_rate))

warmup_gpu_index = gpus.split(",")[0]
max_batches = max_batches // len(gpus.split(","))
if max_batches < warmup_iterations:
    max_batches = warmup_iterations

# start watcher
watcher = train_watcher.TrainWatcher(model_dir='/out/models/',
                                     width=image_width,
                                     height=image_height,
                                     class_num=classnum)
watcher.start()

pretrained_model_params = None
if pretrained_model_params_conf:
    if isinstance(pretrained_model_params_conf, list):
        # list
        for model_path in pretrained_model_params_conf:
            if os.path.splitext(model_path)[1] == '.weights':
                pretrained_model_params = model_path
                break

        if not pretrained_model_params:
            raise ValueError("can not find proper pretrained model in config: {}".format(pretrained_model_params_conf))
    else:
        # other types: not supported
        raise ValueError("unsupported pretrained_model_params_list: {}".format(type(pretrained_model_params_conf)))

# run training
if pretrained_model_params is None or not os.path.isfile(pretrained_model_params):
    # if pretrained model params doesn't exist, train model from image net pretrain model
    os.system("sed -i 's/max_batches=20000/max_batches={}/g' /out/models/yolov4.cfg".format(warmup_iterations))
    warmup_train_script_str = "./darknet detector train /out/coco.data /out/models/yolov4.cfg ./yolov4.conv.137 -map -gpus {} -task_id {} -max_batches {} -dont_show".format(warmup_gpu_index, task_id, max_batches)
    logging.info(f"warmup: {warmup_train_script_str}")
    os.system("python3 warm_up_training.py --train_script='{}' --gpus='{}'".format(warmup_train_script_str, gpus))
    time.sleep(60)
    os.system("sed -i 's/max_batches={}/max_batches={}/g' /out/models/yolov4.cfg".format(warmup_iterations, max_batches))
    best_map = get_pretrained_best_map()
    train_script_str = "./darknet detector train /out/coco.data /out/models/yolov4.cfg /out/models/yolov4_last.weights -map -gpus {} -task_id {} -max_batches {} -best_map {} -dont_show".format(gpus, task_id, max_batches, best_map)
else:
    # if pretrained model params does exist, train model from last best weights, clear previous trained count
    model_seen = get_model_seen(pretrained_model_params)
    model_trained_batches = model_seen // batch
    logging.info(f"model already seen: {model_seen}, {model_trained_batches}")
    max_batches += model_trained_batches
    logging.info(f"max_batches reset to {max_batches}")
    os.system("sed -i 's/max_batches=20000/max_batches={}/g' /out/models/yolov4.cfg".format(max_batches))
    train_script_str = "./darknet detector train /out/coco.data /out/models/yolov4.cfg {} -map -gpus {} -task_id {} -max_batches {} -dont_show".format(pretrained_model_params, gpus, task_id, max_batches)

logging.info(f"training: {train_script_str}")
os.system(train_script_str)

best_param_name = "/out/models/yolov4_best.weights"
if not os.path.isfile(best_param_name):
    best_param_name = "/out/models/yolov4_last.weights"

if not os.path.isfile(best_param_name):
    raise FileNotFoundError("cannot find model weight")

# convert model darkent to mxnet
darknet2mxnet_script_str = "python3 convert_model_darknet2mxnet_yolov4.py --input_h={} --input_w={} --num_of_classes={} --load_param_name={}".format(image_height, image_width, classnum, best_param_name)
logging.info(f"converting: {darknet2mxnet_script_str}")
os.system(darknet2mxnet_script_str)

# run map and output log
run_map_script_str = "./darknet detector map /out/coco.data /out/models/yolov4.cfg {}".format(best_param_name)
os.system(run_map_script_str)

watcher.stop()
