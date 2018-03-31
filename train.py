import tensorflow as tf
from tensorpack import *
from pathlib import Path
from modules import VGGBlock_ours as VGGBlock, Stage1Block, StageTBlock
from reader import Data
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
import multiprocessing
from cfgs.config import cfg
from utils.session_init import get_model_loader_from_vgg as _get_model_loader

import os
h, w = cfg.crop_size_y, cfg.crop_size_x
stride = cfg.stride
stages = cfg.stages

"""
COCO数据集中
train2014  : 82783 个样本
val2014    : 40504 个样本
数据集生成的时候，将val2014中的前2644个样本标记位 'isValidation = 1'
所以用于训练的样本数为 82783+40504-2644 = 120643
这里设置，我们训练model的总样本数为 6000000
"""

def apply_mask(t, mask):
    return t * mask

def image_preprocess(image, bgr = True):
    with tf.name_scope('image_preprocess'):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)

        mean = [0.485, 0.456, 0.406]    # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image

class Model(ModelDesc):

    
    def __init__(self, mode = 'train'):
        self.is_train = mode == 'train'
        self.apply_mask = self.is_train


    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, None, None, 3), 'imgs'),
            InputDesc(tf.float32, (None, None, None, 19), 'gt_heatmaps'),
            InputDesc(tf.float32, (None, None, None, 38), 'gt_pafs'),
            InputDesc(tf.float32, (None, None, None, 1), 'mask')
        ]


    def _build_graph(self, inputs):
        """
        inputs: (b, h, w, c)
        outputs:
            - part_confidence_maps (b, h, w, c')
            - part_affinity_fields (b, h, w, 2c')
        """
        with tf.device('/gpu:1'):
            imgs, gt_heatmaps, gt_pafs, mask = inputs#, mask_heatmaps, mask_pafs = inputs
    
            # ========================== Preprocess ==========================
            imgs = image_preprocess(imgs, bgr = False)

            # ========================== Forward ==========================
            heatmap_outputs, paf_outputs = [], []
            vgg_output = VGGBlock('VGG', imgs)

            # Stage 1
            branch1, branch2 = Stage1Block('Stage1', vgg_output, 1), Stage1Block('Stage1', vgg_output, 2)
            l = tf.concat([branch1, branch2, vgg_output], axis = -1)

            if self.apply_mask:
                w1 = apply_mask(branch1, mask)
                w2 = apply_mask(branch2, mask)
                heatmap_outputs.append(w1)
                paf_outputs.append(w2)
            else:
                heatmap_outputs.append(branch1)
                paf_outputs.append(branch2)

            # Stage T
            for i in range(2, stages + 1):
                branch1, branch2 = StageTBlock('Stage{}'.format(i), l, 1), StageTBlock('Stage{}'.format(i), l, 2)
                l = tf.concat([branch1, branch2, vgg_output], axis = -1)
                
                if self.apply_mask:
                    w1 = apply_mask(branch1, mask)
                    w2 = apply_mask(branch2, mask)
                    heatmap_outputs.append(w1)
                    paf_outputs.append(w2)
                else:
                    heatmap_outputs.append(branch1)
                    paf_outputs.append(branch2)


            # ========================== Cost Functions ==========================
            loss_total = 0
            loss1_total = 0
            loss2_total = 0
            batch_size = tf.shape(imgs)[0]

            heat_weight = 46 * 46 * 19 / 2.0 # for convenient to compare with origin code
            vec_weight = 46 * 46 * 38 / 2.0
            for heatmaps, pafs in zip(heatmap_outputs, paf_outputs):
                # tf.losses.mean_squared_error是对所有element求平均
                loss1 = tf.losses.mean_squared_error(gt_heatmaps, heatmaps)# * heat_weight
                loss2 = tf.losses.mean_squared_error(gt_pafs, pafs)# * vec_weight
                # loss1 = tf.nn.l2_loss((gt_heatmaps - heatmaps)) / 1
                # loss2 = tf.nn.l2_loss((gt_pafs - pafs)) / 1
                loss1_total += loss1
                loss2_total += loss2
                loss_total += (loss1 + loss2)

            if cfg.weight_decay > 0:
                wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
            else:
                wd_cost = tf.constant(0.0)
            self.cost = tf.add_n([loss_total, wd_cost], name='cost')


            # ========================== Summary & Outputs ==========================
            tf.summary.image(name = 'Image', tensor = imgs, max_outputs=3)
            tf.summary.image(name = 'Mask', tensor = mask, max_outputs=3)
            output1 = tf.identity(heatmap_outputs[-1],  name = 'HeatMaps')
            output2 = tf.identity(paf_outputs[-1], name = 'PAFs')


            add_moving_summary(self.cost)
            add_moving_summary(tf.identity(loss1_total, name = 'HeatMapLoss'))
            add_moving_summary(tf.identity(loss2_total, name = 'PAFLoss'))

            # ht = tf.reduce_sum(gt_heatmaps, axis = -1)
            # xht = tf.reduce_sum(heatmap_outputs[-1], axis = -1)
            # ht = tf.split(gt_heatmaps, 19, axis = -1)[0]
            # xht = tf.split(heatmap_outputs[-1], 19, axis = -1)[0]
            # tf.summary.image(name = 'GT_HeatMap', tensor = ht, max_outputs=3)
            # tf.summary.image(name = 'HeatMap', tensor = xht, max_outputs=3)
        

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary = True)
        
        return tf.train.MomentumOptimizer(learning_rate = lr, momentum = cfg.momentum)


def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'

    ds = Data(train_or_test, True)

    if is_train:
        augmentors = [
            # imgaug.RandomCrop(crop_shape=cfg.img_size),
            # imgaug.RandomOrderAug(
            #     [imgaug.Brightness(30, clip=False),
            #      imgaug.Contrast((0.8, 1.2), clip=False),
            #      imgaug.Saturation(0.4),
            #      # rgb-bgr conversion
            #      imgaug.Lighting(0.1,
            #                      eigval=[0.2175, 0.0188, 0.0045][::-1],
            #                      eigvec=np.array(
            #                          [[-0.5675, 0.7192, 0.4009],
            #                           [-0.5808, -0.0045, -0.8140],
            #                           [-0.5836, -0.6948, 0.4203]],
            #                          dtype='float32')[::-1, ::-1]
            #                      )]),
            # imgaug.Clip(),
            # imgaug.Flip(horiz=True),
            # imgaug.ToUint8()
        ]
    # else:
        # augmentors = [
        #     imgaug.RandomCrop(crop_shape=cfg.img_size),
        #     imgaug.ToUint8()
        # ]
    # ds = AugmentImageComponent(ds, augmentors)

    if is_train:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder = not is_train)
    return ds


def get_config(args):
    dataset_train = get_data('train', int(args.batch_size))
    dataset_val = get_data('test', int(args.batch_size))

    return TrainConfig(
        dataflow = dataset_train,
        callbacks = [
            ModelSaver(),
            PeriodicTrigger(InferenceRunner(dataset_val, [
                ScalarStats('cost')]), every_k_epochs = 3),
            # HumanHyperParamSetter('learning_rate'),
        ],
        model = Model(),
        # steps_per_epoch = 200,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size', help='load model', default = 64)
    parser.add_argument('--log_dir', help="directory of logging", default=None)
    args = parser.parse_args()
    if args.log_dir != None:
        logger.set_logger_dir(str(Path('train_log')/args.log_dir))
    else:
        logger.auto_set_dir()

    config = get_config(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        NR_GPU = len(args.gpu.split(','))
        config.nr_tower = NR_GPU
    if args.load:
        config.session_init = _get_model_loader(args.load)
    
    if args.load:
        if args.load.endswith('.npz'):
            config.session_init = _get_model_loader(args.load)
    trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
    launch_train_with_config(config, trainer)
