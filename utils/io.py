from pathlib import Path
import imageio
import sys
sys.path.insert(1, '../data/coco/PythonAPI/')
from pycocotools.coco import COCO
from cfgs.config import cfg
# from .preprocess import anno_to_ours, compute_neck, gen_heatmaps, gen_pafs
import numpy as np
import os

coco_to_ours = cfg.coco_to_ours


def imread(img_name):
    dataset_name = img_name.split('_')[1]
    dir_name = cfg.dataset_image_dirs[dataset_name]
    img = imageio.imread(os.path.join(dir_name, img_name))
    if img.ndim == 2:
        img = np.stack([img]*3, axis = -1)
    return img


# def load_from_anno(anno_path, img_dir, img_id):
#     coco = COCO(anno_path)
#     ann_ids = coco.getAnnIds(imgIds = img_id)
#     img_dict = coco.imgs[img_id]
#     img_anns = coco.loadAnns(ann_ids)

#     img = imageio.imread(str(Path(img_dir) / img_dict['file_name']))
#     persons = anno_to_ours(img_anns)
#     # persons = anno_to_ours(img_anns)
#     all_keypoints = []
#     for idx, person in enumerate(persons):
#         keypoints = np.zeros((18,3)).tolist()
#         neck = compute_neck(person['keypoints'][5], person['keypoints'][6])
#         keypoints = np.stack(person['keypoints']+[neck])
#         # 交换part的顺序和y,x的顺序
#         keypoints = keypoints[coco_to_ours, :][:, [1, 0, 2]]
#         all_keypoints.append(keypoints)
    

#     heatmaps = gen_heatmaps(img, all_keypoints, 1, th1)
#     pafs = gen_pafs(img, all_keypoints, 1, th2)
#     return [img, heatmaps, pafs]
