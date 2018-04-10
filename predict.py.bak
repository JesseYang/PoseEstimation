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

from utils.postprocess import get_peaks, get_connections, multi_person_parse
from utils.visualize import visualize_heatmaps, visualize_matchsticks
from utils.new_preprocess import pad_right_down_corner

def _predict(img, pred_func):
    stride = cfg.stride

    multiplier = [x * 368 / img.shape[1] for x in [0.5, 1, 1.5, 2]]

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
    paf_avg = np.zeros((img.shape[0], img.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        # CUBIC插值缩放img
        resized_img = cv2.resize(img, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
        # 在右方/下方加padding
        # pdb.set_trace()
        padded_img, pad = pad_right_down_corner(resized_img, 8, 0)
        input_img = np.float32(padded_img[np.newaxis,:,:,:])
        print('========== 输入 ==========')
        print(input_img.shape)
        print('还OK')
        heatmap, paf = pred_func(input_img)
        print('========== 输出 ==========')
        print('HeatMap:', heatmap.shape)
        print('PAF:', paf.shape)
        
        
        # 恢复原大小, 去掉padding
        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(heatmap)
        print(heatmap.shape)
        print(heatmap)
        # heatmap = cv2.resize(heatmap, (0,0), fx = stride, fy = stride, interpolation = cv2.INTER_CUBIC)
        heatmap = heatmap[:padded_img.shape[0]-pad[2], :padded_img.shape[1]-pad[3], :]
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        
        paf = np.squeeze(paf) # output 0 is PAFs
        paf = cv2.resize(paf, (0,0), fx = stride, fy = stride, interpolation = cv2.INTER_CUBIC)
        paf = paf[:padded_img.shape[0]-pad[2], :padded_img.shape[1]-pad[3], :]
        paf = cv2.resize(paf, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
        
        print('HeatMap:', heatmap.shape)
        print('PAF:', paf.shape)
        print('\n'*3)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)
    
    return heatmap_avg, paf_avg



def predict(args):
    # prepare predictor
    sess_init = SaverRestore(args.model_path)
    model = Model('test')
    predict_config = PredictConfig(session_init = sess_init,
                                   model = model,
                                   input_names = ['imgs'],
                                   output_names = ['HeatMaps', 'PAFs'])
    predict_func = OfflinePredictor(predict_config)

    img = cv2.imread(args.input_path)

    img = cv2.resize(img, (cfg.img_y, cfg.img_x))

    import pdb
    pdb.set_trace()

    img = np.expand_dims(img, axis=0)

    heatmap, paf = predict_func(img)

    # heatmap, paf = _predict(img, predict_func)

    # print(heatmap, paf)


if __name__ == '__main__':

    img_id = 262145
    img_path = os.path.join('coco/train2017', '%012d.jpg' % img_id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data', default=img_path)
    args = parser.parse_args()

    predict(args)
