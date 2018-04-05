import tensorflow as tf
from tensorpack import *

from cfgs.config import cfg

# @layer_register(log_shape=True)
def VGGBlock_official(l):
    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
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


# @layer_register(log_shape=True)
def VGGBlock_ours(l):
    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
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
             .MaxPooling('pool3', 2)
             .Conv2D('conv4_1', 512)
             .Conv2D('conv4_2', 512)
             .Conv2D('conv4_3', 512)())
    
    return l


@layer_register(log_shape=True)
def Stage1Block(l, branch):
    assert branch in [1, 2]
    
    ch_out = cfg.ch_heats if branch == 1 else cfg.ch_vectors

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .Conv2D('conv1', 128, 3)
                 .Conv2D('conv2', 128, 3)
                 .Conv2D('conv3', 128, 3)
                 .Conv2D('conv4', 512, 1)
                 .Conv2D('conv5', ch_out, 1)())
                 # .Conv2D('conv5', ch_out, 1, nl=tf.identity)())

    return l


@layer_register(log_shape=True)
def StageTBlock(l, branch):
    assert branch in [1, 2]
    
    ch_out = cfg.ch_heats if branch == 1 else cfg.ch_vectors

    with tf.variable_scope('branch_%d' % branch):
        with argscope(Conv2D, W_init=tf.random_normal_initializer(stddev=0.01), nl=tf.nn.relu):
            l = (LinearWrap(l)
                 .Conv2D('conv1', 128, 7)
                 .Conv2D('conv2', 128, 7)
                 .Conv2D('conv3', 128, 7)
                 .Conv2D('conv4', 128, 7)
                 .Conv2D('conv5', 128, 7)
                 .Conv2D('conv6', 128, 1)
                 .Conv2D('conv7', ch_out, 1)())
                 # .Conv2D('conv7', ch_out, 1, nl=tf.identity)())

    return l
