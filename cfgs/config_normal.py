from easydict import EasyDict as edict


_ = cfg = edict()
_.train_ann = 'coco/annotations/person_keypoints_train2017.json'
_.val_ann = 'coco/annotations/person_keypoints_val2017.json'


_.train_images_dir = 'coco/train2017'
_.val_images_dir = 'coco/val2017'
_.test_images_dir = 'coco/test2017'

_.train_masks_dir = 'coco/train2017_masks'
_.val_masks_dir = 'coco/val2017_masks'


_.train_labels_dir = 'coco/train2017_labels'
_.val_labels_dir = 'coco/val2017_labels'

_.img_y = 368
_.img_x = 368

# whether skip a person if the distance to existing person is too small
_.skip_adj = False

_.augmentation = True
_.backbone_grad_scale = 0.25

_.debug = False
_.debug_sample_num = 1600

# the default oder in coco annotation is:
# 0:nose          1:left_eye         2:right_eye         3:left_ear
# 4:right_ear     5:left_shoulder    6:right_shoulder    7:left_elbow
# 8:right_elbow   9:left_wrist       10:right_wrist      11:left_hip
# 12:right_hip    13:left_knee       14:right_knee       15:left_ankle
# 16:right_ankle
_.from_body_part = [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
_.to_body_part =   [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
# the oder after adjustment is:
# 0:nose         1:neck           2:right_shoulder   3:right_elbow
# 4:right_wrist  5:left_shoulder  6:left_elbow       7:left_wrist
# 8:right_hip    9:right_knee     10:right_anckle    11:left_hip
# 12:left_knee   13:left_ankle    14:right_eye       15:left_eye
# 16:right_ear   17:left_ear

# the limbs include:
# 0:neck-->right_hip          1:right_hip-->right_knee      2:right_knee-->right_ankle      3:neck-->left_hip
# 4:left_hip-->left_knee      5:left_knee-->left_ankle      6:neck-->right_shoulder         7:right_shoulder-->right_elbow
# 8:right_elbow-->right_wrist 9:right_shoulder-->right_ear  10:neck-->left_shoulder         11:left_shoulder-->left_elbow
# 12:left_elbow-->left_wrist  13:left_shoulder-->left_ear   14:neck-->nose                  15:nose-->right_eye
# 16:nose-->left_eye          17:right_eye-->right_ear      18:left_eye-->left_ear
_.limb_from = [1, 8,  9,  1, 11, 12, 1, 2, 3,  2, 1, 5, 6,  5, 1,  0,  0, 14, 15]
_.limb_to =   [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]

_.stride = 8

_.sigma = 7

_.grid_y = int(_.img_y / _.stride)
_.grid_x = int(_.img_x / _.stride)

_.thre = 1

_.ch_heats = 18 + 1 # 18个parts 第19个(遍历不会执行到)表示是非人体区域
_.ch_vectors = 38
_.stages = 6
# ============================================================
#                       预处理
# ============================================================
_ = cfg
# 模型先经过VGG-19的前10层, 有3次MaxPooling, 所以在预处理阶段要将groundtruth缩小8倍
# cfg.stride = 8
# 由KeyPoints生成HeatMap时, Peak的扩散系数
# cfg.gen_heatmap_th = 1
# 由KeyPoints生成PAF时, limb的宽度
# cfg.gen_paf_th = 40

# cfg.crop_size_x = 368
# cfg.crop_size_y = 368

cfg.base_lr = 4e-6
cfg.momentum = 0.9
cfg.weight_decay = 5e-4
# ============================================================
#                       后处理
# ============================================================
cfg.th1 = 0
cfg.th2 = 1

# ==================== Datasets ====================
cfg.anno_paths = [
    'data/coco/annotations/person_keypoints_train2014.json',
    'data/coco/annotations/person_keypoints_val2014.json'
]


cfg.dataset_image_dirs = {
    'train2014': 'data/coco/images/train2014/',
    'val2014': 'data/coco/images/val2014/',
}


cfg.train_list = ['data_train.txt']
cfg.test_list = ['data_test.txt']

# ============================================================
#                       可视化
# ============================================================
cfg.stickwidth = 4
cfg.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [0, 0, 0]]

part_str = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri', 'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear', 'pt19']

cfg.coco_to_ours = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
cfg.ours_to_coco = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10, 1]
cfg.limb_seq = [
    (1,8),   # 脖子->右髋
    (8,9),   # 右髋->右膝
    (9,10),  # 右膝->右脚踝

    (1,11),  # 脖子->左髋
    (11,12), # 左髋->左膝
    (12,13), # 左膝->左脚踝

    (1,2),   # 脖子->右肩
    (2,3),   # 右肩->右肘
    (3,4),   # 右肘->右腕
    (2,16),  # 右肩->右耳

    (1,5),   # 脖子->左肩
    (5,6),   # 左肩->左肘
    (6,7),   # 左肘->左腕
    (5,17),  # 左肩->左耳

    (1,0),   # 脖子->鼻子
    (0,14),  # 鼻子->右眼
    (0,15),  # 鼻子->左眼
    (14,16), # 右眼->右耳
    (15,17), # 左眼->左耳
]
