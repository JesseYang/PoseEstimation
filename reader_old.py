from pathlib import Path
import sys
sys.path.insert(1, '../data/coco/PythonAPI/')
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
# from utils.preprocess import random_crop_and_pad_bottom_right, anno_to_ours, gen_mask, gen_heatmaps, gen_pafs, compute_neck, generate_heatmap

from utils.new_preprocess import random_crop_and_pad_bottom_right, persons_to_keypoints, gen_mask, gen_heatmap, gen_paf, get_coords, anno_to_ours
from utils import io
coco_to_ours = cfg.coco_to_ours
limb_seq = cfg.limb_seq
stride = cfg.stride

cropped_shape = (368, 368)

def read_data(coco, img_id):
    """
    # Arguments
        coco: COCO对象
        img_id: img_id
    """
    img_dict = coco.imgs[img_id]
    img_anns = coco.loadAnns(coco.getAnnIds(imgIds = img_id))

    persons = anno_to_ours(img_anns)
    if len(persons) == 0: return


    # 读取图片
    img_name = coco.imgs[img_id]['file_name']
    img = io.imread(img_name)
    # 生成mask
    mask_all, mask_miss = gen_mask(img, coco, img_anns)
    # 随机crop, pad
    img, mask_all, info = random_crop_and_pad_bottom_right(img, mask_all, (368, 368))
    mask_all = cv2.resize(mask_all, None, fx = 1 / stride, fy = 1 / stride)
    # 生成Keypoints
    keypoints = persons_to_keypoints(persons)
    h_start, w_start = info[:2]
    # 生成coords, 大小为46*46*2, 每个位置是对应的坐标, 以便加速后续的操作
    coords = get_coords((h_start // stride, h_start // stride + 46), (w_start // stride, w_start // stride + 46))
    # 生成HeatMap
    heatmap = gen_heatmap(coords, keypoints, stride, cfg.gen_heatmap_th)
    # 生成PAF
    paf = gen_paf(coords, keypoints, stride, cfg.gen_paf_th)


    # 参考: https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/preprocessing/generate_json_mask.py
    # 这个repo只保存了mask_miss ?
    # 按个人理解, 还是应该输出mask_all的
    # mask_all: 将每个人的segmentation标注(格式是多边形的每个点坐标), 转换为二值mask, 存在标注的keypoints肯定在Segmentation的范围内, 所以滤去mask以外的loss可以说得通
    #           但是从loss上讲, 模型没有学习没有人的地方的结果为背景(heatmap_idx=19)这件事, 而是通过后处理的手段滤去了
    # mask_miss: 可见get-started/preprocess.ipynb, 也是人, 但是is_crowd标注为1, 直观来看就是图中较小的人
    return img, heatmap, paf, mask_all[:,:,np.newaxis]
    




    # ann_ids = coco.getAnnIds(imgIds = img_id)
    # img_dict = coco.imgs[img_id]
    # img_anns = coco.loadAnns(ann_ids)

    # persons = anno_to_ours(img_anns)

    # if len(persons) == 0: return

    # # 读取图片并crop
    # if img_dict['file_name'].find('train2014') > -1:
    #     raw_img = misc.imread('data/coco/images/train2014/' + img_dict['file_name'])
    # else:
    #     raw_img = misc.imread('data/coco/images/val2014/' + img_dict['file_name'])
    # if raw_img.ndim == 2:
    #     raw_img = np.stack([raw_img]*3, axis = -1)

    # mask_all, mask_miss = gen_mask(raw_img, coco, img_anns)

    # img, mask_all, info = random_crop_and_pad_bottom_right(raw_img, mask_all, (368, 368))
    # # img = raw_img[0+300:369+300, 0:369]
    # # mask_all = mask_all[0+300:369+300, 0:369]
    # mask_all = cv2.resize(mask_all, None, fx = 1 / stride, fy = 1 / stride)

    # all_keypoints = []
    # for idx, person in enumerate(persons):
    #     keypoints = np.zeros((18,3)).tolist()
    #     neck = compute_neck(person['keypoints'][5], person['keypoints'][6])
    #     keypoints = np.stack(person['keypoints']+[neck])
    #     # 交换part的顺序和y,x的顺序
    #     keypoints = keypoints[coco_to_ours, :][:, [1, 0, 2]]
    #     all_keypoints.append(keypoints)

    # # heatmaps = gen_heatmaps(img, info, all_keypoints, 8, th1)
    # ht = np.zeros((46, 46, 19))
    # heatmaps = generate_heatmap(ht, all_keypoints, 8, 1)
    # pafs = gen_pafs(img, info, all_keypoints, 8, th2)

    # # TODO: 返回mask
    # return [img, heatmaps, pafs, mask_all[:,:,np.newaxis]]


class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle):
        super(Data, self).__init__()

        assert train_or_test in ['train', 'test']

        # Load Annotations
        self.d = {}
        for anno_path in cfg.anno_paths:
            ann_name = anno_path.split('/')[-1]
            coco = COCO(anno_path)
            self.d[ann_name] = coco
        
        path_list = cfg.train_list if train_or_test == 'train' else cfg.test_list
        content = []
        for path in path_list:
            with open(path, 'r') as f:
                content.extend(f.readlines())

        # Load Paths
        self.data = [x.strip() for x in content]

        self.shuffle = shuffle


    def size(self):
        return len(self.data)


    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.data)

        for line in self.data:
            anno_name, img_id = line.split(',')
            data = read_data(self.d[anno_name], int(img_id))
            # print(data)
            if data is None: continue
            yield data


    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    ds = Data('train', False)
    
    g = ds.get_data()
    for i in g:
        img, heatmaps, pafs, mask_all = i
        print(mask_all.shape)
        # print(np.min(i[1]), np.max(i[1]), np.min(i[2]), np.max(i[2]))
