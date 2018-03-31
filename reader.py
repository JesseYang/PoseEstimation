from pathlib import Path
import pickle
import os
import sys
sys.path.insert(1, '../coco/PythonAPI/')
from pycocotools.coco import COCO

from tensorpack import *
import numpy as np
from scipy import misc
import cv2
import json
from itertools import chain
import math
from cfgs.config import cfg
import random
import time

from utils.new_preprocess import random_crop_and_pad_bottom_right, persons_to_keypoints, gen_mask, gen_heatmap, gen_paf, get_coords, anno_to_ours
from utils import io
coco_to_ours = cfg.coco_to_ours
limb_seq = cfg.limb_seq
stride = cfg.stride

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return np.sqrt(self.x * self.x + self.y * self.y)

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle):
        super(Data, self).__init__()

        assert train_or_test in ['train', 'test']

        self.anno_path = cfg.train_ann if train_or_test == 'train' else cfg.val_ann
        self.labels_dir = cfg.train_labels_dir if train_or_test == 'train' else cfg.val_labels_dir
        self.masks_dir = cfg.train_masks_dir if train_or_test == 'train' else cfg.val_masks_dir
        self.images_dir = cfg.train_images_dir if train_or_test == 'train' else cfg.val_images_dir
        coco = COCO(self.anno_path)

        img_id_list = coco.imgs.keys()
        self.img_id_list = [e for e in list(img_id_list) if os.path.isfile(os.path.join(self.labels_dir, "label_%012d" % e)) == True]

        self.shuffle = shuffle


    def size(self):
        return len(self.img_id_list)


    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.img_id_list)

        for img_id in self.img_id_list:

            # read img, mask, and label data
            img_path = os.path.join(self.images_dir, '%012d.jpg' % img_id)
            img = cv2.imread(img_path)

            mask_path = os.path.join(self.masks_dir, "mask_miss_%012d.png" % img_id)
            mask = cv2.imread(mask_path, 0)

            label_path = os.path.join(self.labels_dir, "label_%012d" % img_id)
            f = open(label_path, 'rb')
            label = pickle.load(f)

            # convert to model input format
            img = cv2.resize(img, (cfg.img_y, cfg.img_x))
            mask = cv2.resize(mask, (cfg.grid_y, cfg.grid_x)) / 255

            # create blank heat map
            heat_maps = np.zeros((cfg.grid_y, cfg.grid_x, cfg.ch_heats))

            # create blank paf
            paf = np.zeros((cfg.grid_y, cfg.grid_x, cfg.ch_vectors))

            start = cfg.stride / 2.0 - 0.5

            for i in range(cfg.ch_heats - 1): # for each keypoint
                for person_label in label:
                    if person_label['joint'][i, 2] > 1: # cropped or unlabeled
                        continue

                    x_center = person_label['joint'][i, 0]
                    y_center = person_label['joint'][i, 1]
                    for g_y in range(cfg.grid_y):
                        for g_x in range(cfg.grid_x):
                            x = start + g_x * cfg.stride
                            y = start + g_y * cfg.stride

                            d2 = (x - x_center) * (x - x_center) + (y - y_center) * (y - y_center)
                            exponent = d2 / 2.0 / cfg.sigma / cfg.sigma
                            if exponent > 4.6052:
                                # //ln(100) = -ln(1%)
                                continue;
                            val = min(np.exp(-exponent), 1)

                            # maximum in original paper, but sum in keras implementation
                            if val > heat_maps[g_y, g_x, i]:
                                heat_maps[g_y, g_x, i] = val

            # the background channel
            for g_y in range(cfg.grid_y):
                for g_x in range(cfg.grid_x):
                    max_val = np.max(heat_maps[g_y, g_x, :])
                    heat_maps[g_y, g_x, cfg.ch_heats - 1] = 1 - max_val

            for i in range(int(cfg.ch_vectors / 2)):
                limb_from_kp = cfg.limb_from[i]
                limb_to_kp = cfg.limb_to[i]
                count = np.zeros((cfg.grid_y, cfg.grid_x))

                for person_label in label:
                    # get keypoint coord in the label map
                    limb_from = Point(x=person_label['joint'][limb_from_kp, 0] / 8,
                                      y=person_label['joint'][limb_from_kp, 1] / 8)
                    limb_from_v = person_label['joint'][limb_from_kp, 2]

                    limb_to = Point(x=person_label['joint'][limb_to_kp, 0] / 8,
                                    y=person_label['joint'][limb_to_kp, 1] / 8)
                    limb_to_v = person_label['joint'][limb_to_kp, 2]

                    if limb_from_v > 1 or limb_to_v > 1:
                        continue

                    bc = Point(x=limb_to.x-limb_from.x,
                               y=limb_to.y-limb_from.y)
                    norm_bc = bc.norm()
                    if norm_bc < 1e-8:
                        continue

                    bc = Point(x=bc.x/norm_bc,
                               y=bc.y/norm_bc)

                    min_x = int(max(round(min(limb_from.x, limb_to.x) - cfg.thre), 0))
                    max_x = int(min(round(max(limb_from.x, limb_to.x) + cfg.thre), cfg.grid_x))

                    min_y = int(max(round(min(limb_from.y, limb_to.y) - cfg.thre), 0))
                    max_y = int(min(round(max(limb_from.y, limb_to.y) + cfg.thre), cfg.grid_y))
                    
                    for g_y in range(min_y, max_y):
                        for g_x in range(min_x, max_x):
                            ba = Point(x=g_x-limb_from.x,
                                       y=g_y-limb_from.y)
                            dist = np.abs(ba.x * bc.y - ba.y * bc.x)

                            if dist > cfg.thre:
                                continue
                            paf[g_y, g_x, i * 2] += bc.x
                            paf[g_y, g_x, i * 2 + 1] += bc.y
                            count[g_y, g_x] += 1

                for g_y in range(0, cfg.grid_y):
                    for g_x in range(0, cfg.grid_x):
                        if count[g_y, g_x] > 0:
                            paf[g_y, g_x, i * 2] = paf[g_y, g_x, i * 2] / count[g_y, g_x]
                            paf[g_y, g_x, i * 2 + 1] = paf[g_y, g_x, i * 2 + 1] / count[g_y, g_x]

            heat_maps = heat_maps * np.expand_dims(mask, axis=2)
            paf = paf * np.expand_dims(mask, axis=2)

            yield [img, heat_maps, paf, mask]


    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    ds = Data('train', False)
    
    g = ds.get_data()
    sample = next(g)
    '''
    for i in g:
        img, heatmaps, pafs, mask_all = i
        print(mask_all.shape)
    '''
