import numpy as np
import random
import math
import sys
sys.path.insert(1, '../')
from cfgs.config import cfg


def random_crop_and_pad_bottom_right(img, mask, target_shape):
    h, w = img.shape[:2]
    target_h, target_w = target_shape
    padding_h, padding_w = 0, 0
    if target_h >= h:
        h_start = 0
        padding_h = target_h - h
        img = np.concatenate([img, np.ones((padding_h, w, 3)) * 125], axis = 0)
        mask = np.concatenate([mask, np.zeros((padding_h, w))], axis = 0)
    else:
        h_start = random.randint(0, h - target_h - 1)

    h, w = img.shape[:2]
    if target_w >= w:
        w_start = 0
        padding_w = target_w - w
        img = np.concatenate([img, np.ones((h, padding_w, 3)) * 125], axis = 1)
        mask = np.concatenate([mask, np.zeros((h, padding_w))], axis = 1)
    else:
        w_start = random.randint(0, w - target_w - 1)
    
    return img[h_start:h_start + target_h, w_start:w_start + target_w], mask[h_start:h_start + target_h, w_start:w_start + target_w], [h_start, w_start, padding_h, padding_w]


def get_coords(h, w):
    """get coords matrix of x
    # Arguments
        h
        w
    
    # Returns
        coords: (h, w, 2)
    """

    # int h, w to (0, h), (0, w)
    if isinstance(h, int):
        h = (0, h)
    if isinstance(w, int):
        w = (0, w)

    h1, h2 = h
    w1, w2 = w
    coords = np.empty((h2-h1, w2-w1, 2), dtype = np.int)
    coords[..., 0] = np.arange(h1, h2)[:, None]
    coords[..., 1] = np.arange(w1, w2)

    return coords


def anno_to_ours(anno):
    """
    anno: coco img_anno
    """

    
    def too_close(person_centers, person_center):
        for pc in person_centers:
            dis = math.sqrt((person_center[0] - pc[0]) * (person_center[0] - pc[0]) + (person_center[1] - pc[1]) * (person_center[1] - pc[1]))
            if dis < pc[2] * 0.3:
                return True
        return False


    num_of_people = len(anno)
    persons, person_centers = [], []
    for p in range(num_of_people):
        # 跳过part过少的人或者分割区域过小的人
        if anno[p]['num_keypoints'] < 5 or anno[p]['area'] < 32 * 32: continue
        kpt = anno[p]['keypoints']
        # 计算 person center
        person_center = [anno[p]['bbox'][0] + anno[p]['bbox'][2] / 2.0, anno[p]['bbox'][1] + anno[p]['bbox'][3] / 2.0]
        # 跳过和已处理的人距离过近的人
        if too_close(person_centers, person_center): continue
        
        scale = anno[p]['bbox'][3] / 368.0

        dic = dict()
        dic['objpos'] = person_center
        dic['keypoints'] = np.zeros((17, 3)).tolist()
        dic['scale'] = scale
        for part in range(17):
            dic['keypoints'][part][0] = kpt[part * 3]
            dic['keypoints'][part][1] = kpt[part * 3 + 1]
            # visiable - 1, unvisiable - 0, not labeled - 2
            if kpt[part * 3 + 2] == 2:
                dic['keypoints'][part][2] = 1
            elif kpt[part * 3 + 2] == 1:
                dic['keypoints'][part][2] = 0
            else:
                dic['keypoints'][part][2] = 2

        persons.append(dic)
        person_centers.append(np.append(person_center, max(anno[p]['bbox'][2], anno[p]['bbox'][3])))
    
    return persons


def persons_to_keypoints(persons):


    def compute_neck(left_shoulder, right_shoulder):
        neck = np.zeros(3)
        # 计算y, x
        neck[0] = (left_shoulder[0] + right_shoulder[0]) / 2
        neck[1] = (left_shoulder[1] + right_shoulder[1]) / 2

        # 可见性的逻辑:
        # COCO中: 1表示进行了标记，并且可见；0表示进行了标记，但是不可见；2表示没有进行标记
        # - 如果左右肩有一个是无标记的, neck也应无标记
        # - 有标记情况, 只有在左右肩均可见的情况下, neck才可见(Official Repo的逻辑)
        # - 都按可见算(https://github.com/raymon-tian/keras_Realtime_Multi-Person_Pose_Estimation/blob/master/train/COCOLmdb.py 的逻辑)
        if left_shoulder[2] == 2 or right_shoulder[2] == 2: neck[2] = 2
        elif left_shoulder[2] == right_shoulder[2] == 1: neck[2] = 1
        else: neck[2] = 0

        return neck

    def change_order(keypoints):
        """
        # 输入
            keypoints: np.ndarray, 18*3, 代表18种关键点(顺序为coco order)的x,y,flag(标记且可见为1, 标记不可见为0, 无标记为2)
        # 输出
            keypoints: np.ndarray, 18*3, 代表18种关键点(顺序为our order)的y, x,flag(标记且可见为1, 标记不可见为0, 无标记为2)
        """
        return keypoints[cfg.coco_to_ours, :][:, [1, 0, 2]]


    all_keypoints = []
    for idx, person in enumerate(persons):
        keypoints = np.zeros((18,3)).tolist()
        neck = compute_neck(person['keypoints'][5], person['keypoints'][6])
        keypoints = np.stack(person['keypoints']+[neck])
        keypoints = change_order(keypoints)
        all_keypoints.append(keypoints)

    return all_keypoints

def gen_mask(img, coco, img_anns):
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

    mask_miss = mask_miss.astype(np.uint8)
    mask_miss *= 255

    return mask_all, mask_miss

def gen_heatmap(coords, all_keypoints, stride, th):
    """
    https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/CocoFolder.py
    这个repo的numpy写法
    """
    H, W = coords.shape[:2]
    
    heatmap = np.zeros((H, W, 19))
    for keypoints in all_keypoints:
        for part_idx in range(18):
            if keypoints[part_idx][2] == 2: continue
            y, x = keypoints[part_idx][:2]
            y, x = y / stride, x / stride
            yy, xx = coords[:, :, 0], coords[:, :, 1]
            dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / th / th

            heatmap[:, :, part_idx] = np.exp(-dis)
            
    heatmap[heatmap > 1] = 1

    return heatmap


def pad_right_down_corner(img, stride, pad_value):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + pad_value, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + pad_value, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def gen_paf(coords, all_keypoints, stride, th):
    """
    https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation/blob/master/CocoFolder.py
    这个repo的numpy写法
    """
    H, W = coords.shape[:2]
    pafs = []

    for limb_idx in range(len(cfg.limb_seq)):

        paf = np.zeros((H, W, 2), dtype = np.float32)
        cnt = np.zeros((H, W), dtype = np.int32)
        part_idx_a, part_idx_b = cfg.limb_seq[limb_idx]

        for keypoints in all_keypoints:
            if keypoints[part_idx_a][2] == 2 or keypoints[part_idx_b][2] == 2: continue

            ay = keypoints[part_idx_a][0] * 1.0 / stride
            ax = keypoints[part_idx_a][1] * 1.0 / stride
            by = keypoints[part_idx_b][0] * 1.0 / stride
            bx = keypoints[part_idx_b][1] * 1.0 / stride

            bax = bx - ax
            bay = by - ay
            norm_ba = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9 # to aviod two points have same position.
            bax /= norm_ba
            bay /= norm_ba

            min_w = max(int(round(min(ax, bx) - th)), 0)
            max_w = min(int(round(max(ax, bx) + th)), W // stride)
            min_h = max(int(round(min(ay, by) - th)), 0)
            max_h = min(int(round(max(ay, by) + th)), H // stride)

            py = coords[:,:,0] - ay
            px = coords[:,:,1] - ax
            mask = np.abs(bay * px - bax * py)[min_h: max_h, min_w: max_w] <= th

            paf[min_h: max_h, min_w: max_w, 0][mask] = (paf[min_h: max_h, min_w: max_w, 0][mask] * cnt[min_h: max_h, min_w: max_w][mask] + bay) / (cnt[min_h: max_h, min_w: max_w][mask] + 1)
            paf[min_h: max_h, min_w: max_w, 1][mask] = (paf[min_h: max_h, min_w: max_w, 1][mask] * cnt[min_h: max_h, min_w: max_w][mask] + bax) / (cnt[min_h: max_h, min_w: max_w][mask] + 1)

            cnt[min_h: max_h, min_w: max_w][mask] += 1
        
        pafs.append(paf)

    pafs = np.concatenate(pafs, axis = -1)
    
    return pafs