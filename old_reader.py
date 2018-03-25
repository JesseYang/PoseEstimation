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
from utils.preprocess import random_crop_and_pad_bottom_right, anno_to_ours, gen_mask, gen_heatmaps, gen_pafs, compute_neck

coco_to_ours = cfg.coco_to_ours
limb_seq = cfg.limb_seq
th1 = cfg.peak_spread_factor
th2 = cfg.limb_width
stride = cfg.stride

cropped_shape = (368, 368)

def read_data(coco, img_id):
    """
    # Arguments
        coco: COCO对象
        img_id: img_id
    """
    ann_ids = coco.getAnnIds(imgIds = img_id)
    img_dict = coco.imgs[img_id]
    img_anns = coco.loadAnns(ann_ids)
    
    persons = anno_to_ours(img_anns)


    if len(persons) > 0:
        # 读取图片并crop
        if img_dict['file_name'].find('train2014') > -1:
            raw_img = misc.imread('data/coco/images/train2014/' + img_dict['file_name'])
        else:
            raw_img = misc.imread('data/coco/images/val2014/' + img_dict['file_name'])
        if raw_img.ndim == 2:
            raw_img = np.stack([raw_img]*3, axis = -1)
        t = time.time()
        img, info = random_crop_and_pad_bottom_right(raw_img, (368, 368))
        # print('RandomCrop:', time.time() - t)

        all_keypoints = []
        for idx, person in enumerate(persons):
            keypoints = np.zeros((18,3)).tolist()
            neck = compute_neck(person['keypoints'][5], person['keypoints'][6])
            keypoints = np.stack(person['keypoints']+[neck])
            # 交换part的顺序和y,x的顺序
            keypoints = keypoints[coco_to_ours, :][:, [1, 0, 2]]
            # 对keypoints进行downscaling, 包括去掉无用的keypoints (crop后在box外面)
            # 似乎PAF这一步还得用?
            # keypoints = downscale(keypoints, stride, info)
            all_keypoints.append(keypoints)
        
        t = time.time()
        heatmaps = gen_heatmaps(img, all_keypoints, 8, th1)
        # print('生成HeatMap:', time.time() - t)
        t = time.time()
        pafs = gen_pafs(img, all_keypoints, 8, th2)
        # print('生成PAF:', time.time() - t)

        # TODO: 返回mask
        return [img, heatmaps, pafs]#, None, None]

    # 随机缩放

    # 随机旋转

class Data(RNGDataFlow):
    def __init__(self, anno_paths, shuffle):
        super(Data, self).__init__()

        if isinstance(anno_paths, str):
            anno_paths = [anno_paths]
        
        self.cocos = []
        self.ids = []

        for idx, anno_path in enumerate(anno_paths):
            coco = COCO(anno_paths[0])
            self.cocos.append(coco)
            self.ids.extend([(idx, i) for i in coco.imgs.keys()])

        # self.ann, self.img_path, self.mask_path = chain(), chain(), chain()

        # for ann_json, filelist_txt, masklist_txt in file_lists:
        #     with open(ann_json, 'r') as f:
        #         ann = json.loads('\n'.join(f.readlines()))
        #         self.ann = chain(self.ann, ann)

        #     with open(filelist_txt, 'r') as f:
        #         img_path = [i[:-1] for i in f.readlines()]
        #         self.img_path = chain(self.img_path, img_path)

        #     with open(masklist_txt, 'r') as f:
        #         mask_path = [i[:-1] for i in f.readlines()]
        #         self.mask_path = chain(self.mask_path, mask_path)
        
        # self.data = list(zip(self.ann, self.img_path, self.mask_path))
        self.shuffle = shuffle


    def size(self):
        return len(self.ids)


    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.ids)

        for dataset_id, img_id in self.ids:
            data = read_data(self.cocos[dataset_id], img_id)
            if data is None: continue
            yield data


    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    # df = Data(cfg.train_list, shuffle = False)
    # print(df.size())
    # df = BatchData(df, 64, remainder = not True)
    # df.reset_state()
    # g = df.get_data()
    # for i in g:
    #     print(i[0].shape, i[1].shape)
    # with open('../data/coco/annotations/person_keypoints_train2014.json', 'r') as f:
    #     annos = json.loads('\n'.join(f.readlines()))
    # coco = COCO('../data/coco/annotations/person_keypoints_train2014.json')
    # ids = list(coco.imgs.keys())
    # print(ids)
    # quit()
    # data_id = 0
    # anno, img = read_data(annos['annotations'], data_id)
    # print(anno, img)

    anno_paths = [
        'data/coco/annotations/person_keypoints_train2014.json',
        'data/coco/annotations/person_keypoints_val2014.json'
    ]
    ds = Data(anno_paths, False)
    # ds2 = BatchData(ds, 8, remainder = False)
    ds.reset_state()
    # print(ds.size(), ds2.size())

    
    g = ds.get_data()
    import time
    t1 = time.time()
    for idx, data in enumerate(g):
        # t2 = time.time()
        # print(t2 - t1)
        # t1 = t2
        pass
        # if data is not None:
        #     img, heatmap, paf = data
        #     for i in data:
        #         if i is not None:
        #             print(i.shape, end = '\t')
        #         else:
        #             print('None', end = '\t')
        #     print('\n')
