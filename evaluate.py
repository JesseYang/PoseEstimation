import math
import matplotlib
import os
import sys
sys.path.insert(1, '../coco/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tqdm import tqdm
import json
import argparse
from tensorpack import *
from operator import itemgetter
from itertools import groupby
import numpy as np
from predict import *
from cfgs.config import cfg
import cv2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required=True)
    parser.add_argument('--result_path', help='path of result json file', required=True)
    args = parser.parse_args()

    coco = COCO(cfg.val_ann)

    if os.path.isfile(args.result_path) == False:
        predict_func = initialize(args.model_path)

        img_id_list = coco.imgs.keys()
        imgs_dir = cfg.val_images_dir

        pred_results = []

        for img_id in tqdm(img_id_list):
            img_path = os.path.join(imgs_dir, '%012d.jpg' % img_id)
            img = cv2.imread(img_path)

            candidate, all_peaks, subset = detect(img, predict_func)

            for person in subset:
                score = person[-2]

                key_points = np.zeros((len(cfg.coco_to_ours), 3))

                for coco_idx, our_idx in enumerate(cfg.coco_to_ours):
                    peak_idx = int(person[our_idx])
                    if peak_idx == -1:
                        continue
                    peak = candidate[peak_idx]
                    key_points[coco_idx, 0] = peak[0]
                    key_points[coco_idx, 1] = peak[1]
                    key_points[coco_idx, 2] = 1

                temp = np.sum(key_points, 0)
                ave_coord = temp[:2] / temp[2]
                for coco_idx in range(len(cfg.coco_to_ours)):
                    if key_points[coco_idx, 2] == 0:
                        key_points[coco_idx, 0] = ave_coord[0]
                        key_points[coco_idx, 1] = ave_coord[1]
                        key_points[coco_idx, 2] = 1

                key_points_list = list(key_points.reshape(-1))
                result = {"image_id": img_id,
                          "category_id": 1,
                          "keypoints": key_points_list,
                          "score": score}

                pred_results.append(result)

        result_json_str = json.dumps(pred_results)
        f = open('output.json', 'w')
        f.write(result_json_str)
        f.close()
        
    coco_det = coco.loadRes(args.result_path)

    img_ids = sorted(coco.getImgIds())
    coco_eval = COCOeval(coco, coco_det, 'keypoints')
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

