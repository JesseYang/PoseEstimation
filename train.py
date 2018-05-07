import os
import multiprocessing
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils import optimizer, gradproc

from cfgs.config import cfg
if cfg.backbone == 'vgg19':
    from modules import VGGBlock as Backbone, Stage1Block, StageTBlock
else:
    # from modules import Mobilenetv2Block as Backbone, Stage1DepthBlock as Stage1Block, StageTDepthBlock as StageTBlock
    from modules import Mobilenetv2Block as Backbone, Stage1Block, StageTBlock
from reader import Data

def apply_mask(t, mask):
    return t * mask

def image_preprocess(image, bgr=True):
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

    def __init__(self, mode='train'):
        self.is_train = mode == 'train'
        self.apply_mask = self.is_train

    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, None, None, 3), 'imgs'),
            InputDesc(tf.float32, (None, None, None, cfg.ch_heats), 'gt_heatmaps'),
            InputDesc(tf.float32, (None, None, None, cfg.ch_vectors), 'gt_pafs'),
            InputDesc(tf.float32, (None, None, None, 1), 'mask')
        ]

    def _build_graph(self, inputs):
        imgs, gt_heatmaps, gt_pafs, mask = inputs
    
        imgs = image_preprocess(imgs, bgr=True)

        heatmap_outputs, paf_outputs = [], []
        vgg_output = Backbone(imgs)

        # vgg_output = tf.stop_gradient(vgg_output)
        vgg_output = tf.identity(vgg_output, name='vgg_features')

        # Stage 1
        branch1, branch2 = Stage1Block('stage_1', vgg_output, 1), Stage1Block('stage_1', vgg_output, 2)
        l = tf.concat([branch1, branch2, vgg_output], axis=-1)

        if self.apply_mask:
            w1 = apply_mask(branch1, mask)
            w2 = apply_mask(branch2, mask)
            paf_outputs.append(w1)
            heatmap_outputs.append(w2)
        else:
            paf_outputs.append(branch1)
            heatmap_outputs.append(branch2)

        # Stage T
        for i in range(2, cfg.stages + 1):
            branch1, branch2 = StageTBlock('stage_{}'.format(i), l, 1), StageTBlock('stage_{}'.format(i), l, 2)
            l = tf.concat([branch1, branch2, vgg_output], axis=-1)
            
            if self.apply_mask:
                w1 = apply_mask(branch1, mask)
                w2 = apply_mask(branch2, mask)
                paf_outputs.append(w1)
                heatmap_outputs.append(w2)
            else:
                paf_outputs.append(branch1)
                heatmap_outputs.append(branch2)


        # ========================== Cost Functions ==========================
        loss1_list = []
        loss2_list = []
        batch_size = tf.shape(imgs)[0]

        for heatmaps, pafs in zip(heatmap_outputs, paf_outputs):
            loss1 = tf.nn.l2_loss((gt_pafs - pafs)) / tf.cast(batch_size, tf.float32)
            loss2 = tf.nn.l2_loss((gt_heatmaps - heatmaps)) / tf.cast(batch_size, tf.float32)
            loss1_list.append(loss1)
            loss2_list.append(loss2)

        if cfg.weight_decay > 0:
            wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        else:
            wd_cost = tf.constant(0.0)
        loss1_total = tf.add_n(loss1_list)
        loss2_total = tf.add_n(loss2_list)
        loss_total = loss1_total + loss2_total
        self.cost = tf.add_n([loss_total, wd_cost], name='cost')


        # ========================== Summary & Outputs ==========================
        tf.summary.image(name='image', tensor=imgs, max_outputs=3)
        tf.summary.image(name='mask', tensor=mask, max_outputs=3)
        output1 = tf.identity(heatmap_outputs[-1],  name = 'heatmaps')
        output2 = tf.identity(paf_outputs[-1], name = 'pafs')

        add_moving_summary(self.cost)
        for idx, loss1 in enumerate(loss1_list):
            add_moving_summary(tf.identity(loss1_list[idx], name='stage%d_L1_loss' % (idx+1)))
        for idx, loss2 in enumerate(loss2_list):
            add_moving_summary(tf.identity(loss2_list[idx], name='stage%d_L2_loss' % (idx+1)))
        add_moving_summary(tf.identity(loss1_total, name = 'L1_loss'))
        add_moving_summary(tf.identity(loss2_total, name = 'L2_loss'))

        gt_joint_heatmaps = tf.split(gt_heatmaps, [18, 1], axis=3)[0]
        gt_heatmap_shown = tf.reduce_max(gt_joint_heatmaps, axis=3, keep_dims=True)
        joint_heatmaps = tf.split(heatmap_outputs[-1], [18, 1], axis=3)[0]
        heatmap_shown = tf.reduce_max(joint_heatmaps, axis=3, keep_dims=True)
        tf.summary.image(name='gt_heatmap', tensor=gt_heatmap_shown, max_outputs=3)
        tf.summary.image(name='heatmap', tensor=heatmap_shown, max_outputs=3)
        

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=cfg.momentum)
        gradprocs = [gradproc.ScaleGradient(
                     [('conv.*/W', 1),
                      ('conv.*/b', cfg.bias_lr_mult),
                      ('bottleneck.*/W', 1),
                      ('bottleneck.*/b', cfg.bias_lr_mult),
                      ('stage_1.*/W', 1),
                      ('stage_1.*/b', cfg.bias_lr_mult),
                      ('stage_[2-6].*/W', cfg.lr_mult),
                      ('stage_[2-6].*/b', cfg.lr_mult * cfg.bias_lr_mult)])]
        return optimizer.apply_grad_processors(opt, gradprocs)


def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'

    ds = Data(train_or_test, True)
    sample_num = ds.size()

    if is_train:
        augmentors = [
            imgaug.RandomOrderAug(
                [imgaug.Brightness(30, clip=False),
                 imgaug.Contrast((0.8, 1.2), clip=False),
                 imgaug.Saturation(0.4),
                 imgaug.Lighting(0.1,
                                 eigval=[0.2175, 0.0188, 0.0045][::-1],
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Clip(),
            imgaug.ToUint8()
        ]

    else:
        augmentors = [
            imgaug.ToUint8()
        ]
    ds = AugmentImageComponent(ds, augmentors)

    if is_train:
        ds = PrefetchDataZMQ(ds, min(8, multiprocessing.cpu_count()))
    ds = BatchData(ds, batch_size, remainder = not is_train)
    return ds, sample_num

def get_config(args, model):
    ds_train, sample_num = get_data('train', args.batch_size_per_gpu)
    ds_val, _ = get_data('test', args.batch_size_per_gpu)

    return TrainConfig(
        dataflow = ds_train,
        callbacks = [
            ModelSaver(),
            # PeriodicTrigger(InferenceRunner(ds_val, [ScalarStats('cost')]),
            #                 every_k_epochs=3),
            HumanHyperParamSetter('learning_rate'),
        ],
        model = model,
        steps_per_epoch = sample_num // (args.batch_size_per_gpu * get_nr_gpu()),
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default='0,1')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch_size_per_gpu', type=int, default=16)
    parser.add_argument('--logdir', help="directory of logging", default=None)
    parser.add_argument('--flops', action="store_true", help="print flops and exit")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model()
    if args.flops:
        output_y = int(cfg.img_y / cfg.stride)
        output_x = int(cfg.img_x / cfg.stride)

        input_desc = [
            InputDesc(tf.float32, (1, cfg.img_y, cfg.img_x, 3), 'imgs'),
            InputDesc(tf.float32, (1, output_y, output_x, cfg.ch_heats), 'gt_heatmaps'),
            InputDesc(tf.float32, (1, output_y, output_x, cfg.ch_vectors), 'gt_pafs'),
            InputDesc(tf.float32, (1, output_y, output_x, 1), 'mask')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()

        config = get_config(args, model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        
        trainer = SyncMultiGPUTrainerParameterServer(get_nr_gpu())
        launch_train_with_config(config, trainer)
