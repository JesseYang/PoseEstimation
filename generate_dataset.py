import os
import cv2
import sys
sys.path.insert(1, '../coco/cocoapi/PythonAPI/')
from scipy.spatial.distance import cdist
import numpy as np
import pickle
from tqdm import tqdm
from pycocotools.coco import COCO
from utils.preprocess import anno_to_ours
import random

from cfgs.config import cfg

def transform_joints(joints):
    # add neck, adjust order, pay attention to the visibility of neck
    transformed_joints = np.zeros((18, 3))

    for idx in range(18):
        transformed_joints[idx, 0] = (joints[cfg.from_body_part[idx], 0] + joints[cfg.to_body_part[idx], 0]) / 2 
        transformed_joints[idx, 1] = (joints[cfg.from_body_part[idx], 1] + joints[cfg.to_body_part[idx], 1]) / 2 

        if joints[cfg.from_body_part[idx], 2] == 3 or joints[cfg.to_body_part[idx], 2] == 3:
            transformed_joints[idx, 2] = 3
        elif joints[cfg.from_body_part[idx], 2] == 2 or joints[cfg.to_body_part[idx], 2] == 2:
            transformed_joints[idx, 2] = 2
        else:
            transformed_joints[idx, 2] = joints[cfg.from_body_part[idx], 2] and joints[cfg.to_body_part[idx], 2]

    return transformed_joints


def load_dataset(ann_path, images_dir, masks_dir, labels_dir):

    coco = COCO(ann_path)
    for img_id in tqdm(coco.imgs.keys(), ascii=True):

        ann_ids = coco.getAnnIds(imgIds = img_id)
        img_dict = coco.imgs[img_id]
        img_anns = coco.loadAnns(ann_ids)
    
        num_people = len(img_anns)
        persons = []
        prev_center = []
    
        for p in range(num_people):
            if img_anns[p]["num_keypoints"] < 5 or img_anns[p]["area"] < 32 * 32:
                continue
    
            anno = img_anns[p]["keypoints"]
            pers = dict()
            person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                             img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]
    
            # skip this person if the distance to existing person is too small
            if cfg.skip_adj == True:
                flag = 0
                for pc in prev_center:
                    a = np.expand_dims(pc[:2], axis=0)
                    b = np.expand_dims(person_center, axis=0)
                    dist = cdist(a, b)[0]
                    if dist < pc[2] * 0.3:
                        flag = 1
                        continue
    
                if flag == 1:
                    continue
    
            pers["objpos"] = person_center
            pers["bbox"] = img_anns[p]["bbox"]
            pers["segment_area"] = img_anns[p]["area"]
            pers["num_keypoints"] = img_anns[p]["num_keypoints"]
    
            joints = np.zeros((17, 3))
            for part in range(17):
                joints[part, 0] = anno[part * 3]
                joints[part, 1] = anno[part * 3 + 1]
    
                # the visible meaning is adjusted to:
                #   0: labeled but not visible
                #   1: labeled and visible
                #   2: cropped
                #   3: not labeled
                if anno[part * 3 + 2] == 2:
                    # labeled and visible
                    joints[part, 2] = 1
                elif anno[part * 3 + 2] == 1:
                    # labeled but not visible
                    joints[part, 2] = 0
                else:
                    # not labeled
                    joints[part, 2] = 3
    
            pers["joint"] = transform_joints(joints)
    
            pers["scale_provided"] = img_anns[p]["bbox"][3] / cfg.img_y
    
            persons.append(pers)
            prev_center.append(np.append(person_center, max(img_anns[p]["bbox"][2], img_anns[p]["bbox"][3])))

        if len(persons) > 0:
            # generate mask image and save
            img_path = os.path.join(images_dir, '%012d.jpg' % img_id)
            img = cv2.imread(img_path)
            h, w, c = img.shape
            mask_all = np.zeros((h, w), dtype=np.uint8)
            mask_miss = np.zeros((h, w), dtype=np.uint8)
    
            flag = 0
            for p in img_anns:
                seg = p["segmentation"]
        
                if p["iscrowd"] == 1:
                    mask_crowd = coco.annToMask(p)
                    temp = np.bitwise_and(mask_all, mask_crowd)
                    mask_crowd = mask_crowd - temp
                    flag += 1
                    continue
                else:
                    mask = coco.annToMask(p)
        
                mask_all = np.bitwise_or(mask, mask_all)
        
                if p["num_keypoints"] <= 0:
                    mask_miss = np.bitwise_or(mask, mask_miss)
        
            if flag < 1:
                mask_miss = np.logical_not(mask_miss)
            elif flag == 1:
                mask_miss = np.logical_not(np.bitwise_or(mask_miss, mask_crowd))
                mask_all = np.bitwise_or(mask_all, mask_crowd)
            else:
                raise Exception("crowd segments > 1")
        
            mask_miss_path = os.path.join(masks_dir, "mask_miss_%012d.png" % img_id)
            mask_all_path = os.path.join(masks_dir, "mask_all_%012d.png" % img_id)
    
            cv2.imwrite(mask_miss_path, mask_miss * 255)
            cv2.imwrite(mask_all_path, mask_all * 255)

            # save label data
            label_path = os.path.join(labels_dir, "label_%012d" % img_id)
            f = open(label_path, 'wb')
            pickle.dump(persons, f)


# load training set
print("Generating training set")
load_dataset(cfg.train_ann, cfg.train_images_dir, cfg.train_masks_dir, cfg.train_labels_dir)
# load val set
print("Generating validation set")
load_dataset(cfg.val_ann, cfg.val_images_dir, cfg.val_masks_dir, cfg.val_labels_dir)

