import tensorflow as tf
from tensorpack import *

from cfgs.config import cfg

# TODO: 每一个conv后好像都要做normlazation, 写成nl的形式更好

@layer_register(log_shape = True)
def VGGBlock_official(l):
    l = (LinearWrap(l)
    .Conv2D('Conv1_1', out_channel = 64, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv1_2', out_channel = 64, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .MaxPooling('MaxPooling1', 2)
    .Conv2D('Conv2_1', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv2_2', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .MaxPooling('MaxPooling2', 2)
    .Conv2D('Conv3_1', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv3_2', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv3_3', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv3_4', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .MaxPooling('MaxPooling3', 2)
    .Conv2D('Conv4_1', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)
    .Conv2D('Conv4_2', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same', nl = tf.nn.relu)())
    
    return l


@layer_register(log_shape = True)
def VGGBlock_ours(l):
    l = (LinearWrap(l)
    .Conv2D('Conv1_1', out_channel = 64, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN1')
    .tf.nn.relu()
    .Conv2D('Conv1_2', out_channel = 64, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN2')
    .tf.nn.relu()
    .MaxPooling('MaxPooling1', 2)
    .Conv2D('Conv2_1', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN3')
    .tf.nn.relu()
    .Conv2D('Conv2_2', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN4')
    .tf.nn.relu()
    .MaxPooling('MaxPooling2', 2)
    .Conv2D('Conv3_1', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN5')
    .tf.nn.relu()
    .Conv2D('Conv3_2', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN6')
    .tf.nn.relu()
    .Conv2D('Conv3_3', out_channel = 256, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN7')
    .tf.nn.relu()
    .MaxPooling('MaxPooling3', 2)
    .Conv2D('Conv4_1', out_channel = 512, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN8')
    .tf.nn.relu()
    .Conv2D('Conv4_2', out_channel = 512, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN9')
    .tf.nn.relu()
    .Conv2D('Conv4_3', out_channel = 512, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm('BN10')
    .tf.nn.relu()())
    
    return l


@layer_register(log_shape = True)
def Stage1Block(l, branch):
    assert branch in [1, 2]
    
    branch_str = 'Branch{}/'.format(branch)
    ch_out = 19 if branch == 1 else 38
    last_nl = tf.nn.sigmoid if branch == 1 else tf.nn.tanh

    l = (LinearWrap(l)
    .Conv2D(branch_str+'Conv1', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN1')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv2', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN2')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv3', out_channel = 128, kernel_shape = (3,3), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN3')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv4', out_channel = 512, kernel_shape = (1,1), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN4')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv5', out_channel = ch_out, kernel_shape = (1,1), stride = 1, padding = 'same')())

    return l


@layer_register(log_shape = True)
def StageTBlock(l, branch):
    assert branch in [1, 2]
    
    branch_str = 'Branch{}/'.format(branch)
    ch_out = 19 if branch == 1 else 38
    last_nl = tf.nn.sigmoid if branch == 1 else tf.nn.tanh

    l = (LinearWrap(l)
    .Conv2D(branch_str+'Conv1', out_channel = 128, kernel_shape = (7,7), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN1')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv2', out_channel = 128, kernel_shape = (7,7), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN2')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv3', out_channel = 128, kernel_shape = (7,7), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN3')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv4', out_channel = 128, kernel_shape = (7,7), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN4')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv5', out_channel = 128, kernel_shape = (7,7), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN5')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv6', out_channel = 128, kernel_shape = (1,1), stride = 1, padding = 'same')
    # .BatchNorm(branch_str+'BN6')
    .tf.nn.relu()
    .Conv2D(branch_str+'Conv7', out_channel = ch_out, kernel_shape = (1,1), stride = 1, padding = 'same')())

    return l
