import functools
import os
from pathlib import Path

import argparse
from tensorpack import *
from operator import itemgetter
from itertools import groupby
import numpy as np
from train import Model
from reader import Data
from cfgs.config import cfg
import imageio
import cv2
import pdb

from reader import Data


def debug(args):

    # prepare predictor
    sess_init = SaverRestore(args.model_path)
    model = Model('train')
    predict_config = PredictConfig(session_init = sess_init,
                                   model = model,
                                   input_names = ['imgs', 'gt_heatmaps', 'gt_pafs', 'mask'],
                                   output_names = ['vgg_features', 'HeatMaps', 'PAFs', 'cost'])
    predict_func = OfflinePredictor(predict_config)

    ds = Data('train', False)

    g = ds.get_data()
    sample = next(g)

    import pdb
    pdb.set_trace()

    sample = [np.expand_dims(e, axis=0) for e in sample]

    vgg_features, heatmap, paf, cost = predict_func(sample)

    import pdb
    pdb.set_trace()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    args = parser.parse_args()

    debug(args)
