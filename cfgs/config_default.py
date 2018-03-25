from easydict import EasyDict as edict

_ = cfg = edict()
_.ch_heats = 18 + 1 # 18个parts 第19个(遍历不会执行到)表示是非人体区域
_.ch_vectors = _.ch_heats * 2
_.stages = 6
# ============================================================
#                       预处理
# ============================================================
_ = cfg
# 模型先经过VGG-19的前10层, 有3次MaxPooling, 所以在预处理阶段要将groundtruth缩小8倍
cfg.stride = 8
# 由KeyPoints生成HeatMap时, Peak的扩散系数
cfg.gen_heatmap_th = 1
# 由KeyPoints生成PAF时, limb的宽度
cfg.gen_paf_th = 40

cfg.crop_size_x = 368
cfg.crop_size_y = 368

cfg.base_lr = 4e-5
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


cfg.train_list = ['data_train.1.txt']
cfg.test_list = ['data_test.1.txt']

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