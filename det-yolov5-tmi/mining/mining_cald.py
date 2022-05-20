import os
import os.path as osp
import numpy as np
import yaml
import time
from tqdm import tqdm
from mining.data_augment import horizontal_flip, cutout, rotate, intersect, resize
# from mmdet.apis import inference_detector, init_detector
import torch
from scipy.stats import entropy
from concurrent.futures import ThreadPoolExecutor
import cv2
from executor import dataset_reader as dr, env, monitor, result_writer as rw

from loguru import logger
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.augmentations import letterbox
from utils.torch_utils import select_device

def init_detector(device):
    executor_config=env.get_executor_config()

    weights = None
    model_params_path = executor_config['model_params_path']
    if 'best.pt' in model_params_path:
        weights = '/in/models/best.pt'
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                weights=f'/in/models/{f}'
                break 
    
    if weights is None:
        weights = 'yolov5s.pt'
        logger.info(f'cannot find pytorch weight in {model_params_path}, use {weights} instead')

    model = DetectMultiBackend(weights=weights,
        device=device,
        dnn=False, # not use opencv dnn for onnx inference
        data='data.yaml') # dataset.yaml path

    return model

class MiningCald():
    def __init__(self):
        executor_config=env.get_executor_config()
        gpu_id=executor_config['gpu_id']
        gpu_num=len(gpu_id.split(','))
        if gpu_num==0:
            device='cpu'
        else:
            device=gpu_id
        device = select_device(device)

        self.model=init_detector(device)
        self.device=device

        self.stride=self.model.stride
        imgsz = (640, 640)
        imgsz = check_img_size(imgsz, s=self.stride)
        # Run inference
        self.model.warmup(imgsz=(1 , 3, *imgsz), half=False)  # warmup

        self.img_size=imgsz

    def predict(self, img):
        # preprocess 
        # img0 = cv2.imread(path)  # BGR
        # Padded resize
        img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

        # Convert
        img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img1 = np.ascontiguousarray(img1)
        img1 = torch.from_numpy(img1).to(self.device)

        img1 = img1/255  # 0 - 255 to 0.0 - 1.0
        if len(img1.shape) == 3:
            img1 = img1[None]  # expand for batch dim

        pred = self.model(img1)

        # postprocess
        conf_thres=0.25
        iou_thres=0.45
        classes=None
        agnostic_nms=False
        max_det=1000
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        logger.info(f'pred={pred}')
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to img size
                det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
        return pred

    def mining(self):
        path_env = env.get_current_env()
        N=dr.dataset_size(env.DatasetType.CANDIDATE)
        idx=0
        for asset_path, _ in tqdm(dr.item_paths(dataset_type=env.DatasetType.CANDIDATE)):
            img_path=osp.join(path_env.input.root_dir, path_env.input.assets_dir, asset_path)
            img = cv2.imread(img_path)
            pred = self.predict(img) 

            idx+=1
            monitor.write_monitor_logger(percent=0.1 + 0.8*idx/N)

def get_img_path(img_file):
    with open(img_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    names = [line.strip() for line in lines]
    return names


def preprocess_frcnn(src):
    img = cv2.resize(src, (1000, 600))
    return img


class FRCNN:
    def __init__(self, device):
        self.net = init_detector("/in/model/model.py", "/in/model/model.pth", device)
        self.net.eval()


def decode_result(prediction):
    # convert model result to ndarray as
    # img_id, cls, max_score, boxes, max_scores, 1-max_scores
    output = []
    for img_id, each_pred in enumerate(prediction):
        for cls_index, each_cls_result in enumerate(each_pred):
            if each_cls_result.shape[0] == 0:
                continue
            for each_result in each_cls_result:
                xmin, ymin, xmax, ymax, max_score = each_result
                output.append([img_id, cls_index, max_score, xmin, ymin, xmax, ymax, max_score])

    if len(output) == 0:
        return None
    output = np.array(output)

    return output


def detect_img(net, load_images):
    tmp_batch = list(map(preprocess_frcnn, load_images))

    batch_predict_result = inference_detector(net, tmp_batch)
    post_process_result = decode_result(batch_predict_result)

    if post_process_result is not None:
        return post_process_result
    else:
        return None


def compute_cald_thread(net, img_names_batch):
    beta = 1.3
    if len(img_names_batch) == 0:
        return []

    batch_score_list = []

    load_images = []
    for each_img_name in img_names_batch:
        img = cv2.imread(each_img_name)
        assert img is not None
        load_images.append(img)

    result_ref = detect_img(net, load_images)

    if result_ref is None:
        batch_score_list += [-beta for _ in range(len(img_names_batch))]
    else:
        all_image_flip = []
        all_image_cut = []
        all_image_rot = []
        all_image_res = []
        all_boxes_flip = []
        all_boxes_cut = []
        all_boxes_rot = []
        all_boxes_res = []
        for img_ind in range(len(load_images)):
            img_result = result_ref[result_ref[:, 0] == img_ind]
            bboxes = img_result[:, 3:7]

            image = load_images[img_ind]

            if len(bboxes) == 0:
                image_flip, bboxes_flip = image, bboxes
                image_cut, bboxes_cut = image, bboxes
                image_rot, bboxes_rot = image, bboxes
                image_res, bboxes_res = image, bboxes
            else:
                image_flip, bboxes_flip = horizontal_flip(image, bboxes)
                image_cut, bboxes_cut = cutout(image, bboxes)
                image_rot, bboxes_rot = rotate(image, bboxes)
                image_res, bboxes_res = resize(image, bboxes)
            all_image_flip.append(image_flip)
            all_image_cut.append(image_cut)
            all_image_rot.append(image_rot)
            all_image_res.append(image_res)
            all_boxes_flip.append(bboxes_flip)
            all_boxes_cut.append(bboxes_cut)
            all_boxes_rot.append(bboxes_rot)
            all_boxes_res.append(bboxes_res)

        result_flip = detect_img(net, all_image_flip)
        result_cut = detect_img(net, all_image_cut)
        result_rot = detect_img(net, all_image_rot)
        result_res = detect_img(net, all_image_res)

        all_result_aug = [(result_flip, all_boxes_flip), (result_cut, all_boxes_cut), (result_rot, all_boxes_rot), (result_res, all_boxes_res)]
        for img_ind in range(len(load_images)):
            consistency = 0
            cls_scores = []
            aug_cls_scores = []
            for result_aug, origin_aug_boxes in all_result_aug:
                if result_aug is None:
                    consistency += beta
                    continue
                origin_output = result_ref[result_ref[:, 0] == img_ind]
                aug_output = result_aug[result_aug[:, 0] == img_ind]
                if len(origin_output) == 0 or len(aug_output) == 0:
                    consistency += beta
                    continue
                consistency_peraug = get_consistency_per_aug(origin_output,
                                                             aug_output,
                                                             origin_aug_boxes[img_ind],
                                                             cls_scores,
                                                             aug_cls_scores)
                consistency += consistency_peraug

            consistency /= len(all_result_aug)
            batch_score_list.append(-consistency)

    return batch_score_list


def _ious(boxes1, boxes2):
    """
    args:
        boxes1: np.array, (N, 4)
        boxes2: np.array, (M, 4)
    return:
        iou: np.array, (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    iner_area = intersect(boxes1, boxes2)
    area1 = area1.reshape(-1, 1).repeat(area2.shape[0], axis=1)
    area2 = area2.reshape(1, -1).repeat(area1.shape[0], axis=0)
    iou = iner_area / (area1 + area2 - iner_area + 1e-14)
    return iou


def split_prediction(outputs):
    # outputs: [0: img_index, 1: max_cls (based on prob), 2: ness * max_score, 3~6: bbox, 7: max_prob]
    clses = outputs[:, 1]
    scores = outputs[:, 2]
    boxes = outputs[:, 3:7]
    max_scores = outputs[:, 7]
    cls_scores = 1 - max_scores
    return clses, scores, boxes, max_scores, cls_scores


def get_consistency_per_aug(origin_output, aug_output, origin_aug_boxes, cls_scores_list, aug_cls_scores_list):
    beta = 1.3
    clses, scores, boxes, max_scores, cls_scores = split_prediction(origin_output)
    clses_aug, scores_aug, boxes_aug, max_scores_aug, cls_scores_aug = split_prediction(aug_output)

    if cls_scores_list == []:
        cls_scores_list.append(cls_scores)
    aug_cls_scores_list.append(cls_scores_aug)

    # cls, max_scores, boxes, quality, cls_scores, cls_max_score
    # if len(origin_output) == 0:
    #     return 2
    ious = _ious(boxes_aug, origin_aug_boxes)  # N, M
    aug_idxs = np.argmax(ious, axis=0)
    consistency_per_aug = 2
    for origin_idx, aug_idx in enumerate(aug_idxs):
        iou = ious[aug_idx, origin_idx]
        if iou == 0:
            consistency_per_aug = min(consistency_per_aug, beta)
        p = cls_scores_aug[aug_idx]
        q = cls_scores[origin_idx]
        m = (p + q) / 2.
        js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
        if js < 0:
            js = 0
        consistency_box = iou
        consistency_cls = 0.5 * (max_scores[origin_idx] + max_scores_aug[aug_idx]) * (1 - js)
        consistency_per_inst = abs(consistency_box + consistency_cls - beta)
        consistency_per_aug = min(consistency_per_aug, consistency_per_inst.item())

    return consistency_per_aug


if __name__ == "__main__":
    path2score = []
    miner = MiningCald()
    miner.mining()

