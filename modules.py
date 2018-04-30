import tensorflow as tf
from tensorpack import *

from cfgs.config import cfg

def BN(x, name):
    return BatchNorm('bn', x)

def BNReLU6(x, name):
    x = BN(x, 'bn')
    return tf.nn.relu6(x, name=name)

@layer_register(log_shape=True)
def DepthConv(x, out_channel, kernel_shape, padding='SAME', stride=1,
              W_init=None, nl=tf.identity):
    in_shape = x.get_shape().as_list()
    in_channel = in_shape[3]
    assert out_channel % in_channel == 0
    channel_mult = out_channel // in_channel

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    kernel_shape = [kernel_shape, kernel_shape]
    filter_shape = kernel_shape + [in_channel, channel_mult]

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    conv = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding=padding, data_format='NHWC')
    return nl(conv, name='output')



# @layer_register(log_shape=True)
def VGGBlock(l):
    with argscope(Conv2D, kernel_shape=3, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
        l = (LinearWrap(l)
             .Conv2D('conv1_1', 64)
             .Conv2D('conv1_2', 64)
             .MaxPooling('pool1', 2)
             .Conv2D('conv2_1', 128)
             .Conv2D('conv2_2', 128)
             .MaxPooling('pool2', 2)
             .Conv2D('conv3_1', 256)
             .Conv2D('conv3_2', 256)
             .Conv2D('conv3_3', 256)
             .Conv2D('conv3_4', 256)
             .MaxPooling('pool3', 2)
             .Conv2D('conv4_1', 512)
             .Conv2D('conv4_2', 512)
             .Conv2D('conv4_3_cpm', 256)
             .Conv2D('conv4_4_cpm', 128)())
    
    return l

def Mobilenetv2Block(l, data_format='NHWC'):

    def bottleneck_v2(l, t, out_channel, stride=1):
        in_shape = l.get_shape().as_list()

        in_channel = in_shape[1] if data_format == "NCHW" else in_shape[3]
        shortcut = l
        l = Conv2D('conv1', l, t*in_channel, 1, nl=BNReLU6)
        l = DepthConv('depthconv', l, t*in_channel, 3, stride=stride, nl=BNReLU6)
        l = Conv2D('conv2', l, out_channel, 1, nl=BN)
        if stride == 1 and out_channel == in_channel:
            l = l + shortcut
        return l

    with argscope([Conv2D, GlobalAvgPooling, BatchNorm], data_format=data_format), \
            argscope([Conv2D], use_bias=False):
        l = Conv2D('covn1', l, 32, 3, stride=2, nl=BNReLU6)
        conv1 = l

        with tf.variable_scope('bottleneck1'):
            l = bottleneck_v2(l, out_channel=16, t=1, stride=1)

        with tf.variable_scope('bottleneck2'):
            for j in range(2):
                with tf.variable_scope('block{}'.format(j)):
                    l = bottleneck_v2(l, out_channel=24, t=6, stride=2 if j == 0 else 1)
        conv4 = l

        with tf.variable_scope('bottleneck3'):
            for j in range(3):
                with tf.variable_scope('block{}'.format(j)):
                    l = bottleneck_v2(l, out_channel=32, t=6, stride=2 if j == 0 else 1)
        conv7 = l

        '''
        return conv7

        conv1_pool = MaxPooling('pool1', conv1, 2, strides=2)
        conv7_upsample = tf.image.resize_bilinear(conv7,
                                                  [tf.shape(conv7)[1] * 2, tf.shape(conv7)[2] * 2],
                                                  name='upsample')
        channel_axis = 1 if data_format == 'NCHW' else 3
        features = tf.concat([conv1_pool, conv4, conv7_upsample], channel_axis, name='backbone_features')
        '''
        conv4_pool = MaxPooling('pool4', conv4, 2, strides=2)
        channel_axis = 1 if data_format == 'NCHW' else 3
        features = tf.concat([conv4_pool, conv7], channel_axis, name='backbone_features')

        return features
        
@layer_register(log_shape=True)
def Stage1Block(l, branch):
    assert branch in [1, 2]
    
    ch_out = cfg.ch_vectors if branch == 1 else cfg.ch_heats

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .Conv2D('conv1', 128, 3)
                 .Conv2D('conv2', 128, 3)
                 .Conv2D('conv3', 128, 3)
                 .Conv2D('conv4', 512, 1)
                 .Conv2D('conv5', ch_out, 1, nl=tf.identity)())

    return l
        
@layer_register(log_shape=True)
def Stage1DepthBlock(l, branch):
    assert branch in [1, 2]
    
    in_shape = l.get_shape().as_list()
    ch_in = in_shape[3]
    ch_out = cfg.ch_vectors if branch == 1 else cfg.ch_heats

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .DepthConv('conv1_depth', ch_in, 3)
                 .Conv2D('conv1', 128, 1)
                 .DepthConv('conv2_depth', 128, 3)
                 .Conv2D('conv2', 128, 1)
                 .DepthConv('conv3_depth', 128, 3)
                 .Conv2D('conv3', 128, 1)
                 .DepthConv('conv4_depth', 512, 1)
                 .Conv2D('conv4', 512, 1)
                 .Conv2D('conv5', ch_out, 1, nl=tf.identity)())

    return l


@layer_register(log_shape=True)
def StageTBlock(l, branch):
    assert branch in [1, 2]
    
    ch_out = cfg.ch_vectors if branch == 1 else cfg.ch_heats

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .Conv2D('conv1', 128, 7)
                 .Conv2D('conv2', 128, 7)
                 .Conv2D('conv3', 128, 7)
                 .Conv2D('conv4', 128, 7)
                 .Conv2D('conv5', 128, 7)
                 .Conv2D('conv6', 128, 1)
                 .Conv2D('conv7', ch_out, 1, nl=tf.identity)())

    return l

@layer_register(log_shape=True)
def StageTDepthBlock(l, branch):
    assert branch in [1, 2]
    
    in_shape = l.get_shape().as_list()
    ch_in = in_shape[3]
    ch_out = cfg.ch_vectors if branch == 1 else cfg.ch_heats

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .DepthConv('conv1_depth', ch_in, 7)
                 .Conv2D('conv1', 128, 1)
                 .DepthConv('conv2_depth', 128, 7)
                 .Conv2D('conv2', 128, 1)
                 .DepthConv('conv3_depth', 128, 7)
                 .Conv2D('conv3', 128, 1)
                 .DepthConv('conv6_depth', 128, 1)
                 .Conv2D('conv6', 128, 1)
                 .Conv2D('conv7', ch_out, 1, nl=tf.identity)())

    return l
