
import numpy as np
import tensorflow as tf
from tensorpack import *

mapper = {
    'conv1_1': 'VGG/Conv1_1',
    'conv1_2': 'VGG/Conv1_2',
    'conv2_1': 'VGG/Conv2_1',
    'conv2_2': 'VGG/Conv2_2',
    'conv3_1': 'VGG/Conv3_1',
    'conv3_2': 'VGG/Conv3_2',
    'conv3_3': 'VGG/Conv3_3',
    'conv4_1': 'VGG/Conv4_1',
    'conv4_2': 'VGG/Conv4_2',
    'conv4_3': 'VGG/Conv4_3',
}


def get_model_loader_from_vgg(filename):
    """
    Get a corresponding model loader by looking at the file name.
    Returns:
        SessInit: either a :class:`DictRestore` (if name ends with 'npy/npz') or
        :class:`SaverRestore` (otherwise).
    """
    if filename.endswith('.npy'):
        assert tf.gfile.Exists(filename), filename
        return DictRestore(np.load(filename, encoding='latin1').item())
    elif filename.endswith('.npz'):
        assert tf.gfile.Exists(filename), filename
        obj = np.load(filename)
        d = _modify_dict(dict(obj))
        return DictRestore(d)
    else:
        return SaverRestore(filename)


def _modify_dict(d_in):
    d_out = {}
    for key, value in d_in.items():
        layer, parameter = key.split('/')
        if layer not in mapper: continue
        layer = mapper[layer]
        d_out[layer+'/'+parameter] = value
    print(d_out.keys())


    return d_out

if __name__ == '__main__':
    get_model_loader_from_vgg('../models/vgg16_ours.npz')



