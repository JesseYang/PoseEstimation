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

def pad_right_down_corner(img, stride, pad_value):
    h, w, _ = img.shape

    pad = 4 * [0]
    pad[2] = 0 if (h % stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + pad_value, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + pad_value, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

def predict(args):
    # prepare predictor
    sess_init = SaverRestore(args.model_path)
    model = Model('test')
    predict_config = PredictConfig(session_init = sess_init,
                                   model = model,
                                   input_names = ['imgs'],
                                   output_names = ['heatmaps', 'pafs'])
    predict_func = OfflinePredictor(predict_config)

    img = cv2.imread(args.input_path)

    h, w, _ = img.shape

    multiplier = [x * cfg.img_y / h for x in cfg.scale_search]

    heatmap_avg = np.zeros((img.shape[0], img.shape[1], 19))
    paf_avg = np.zeros((img.shape[0], img.shape[1], 38))

    for m in range(len(multiplier)):
        scale = multiplier[m]
        scale_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scale_img_padded, pad = pad_right_down_corner(scale_img, cfg.stride, cfg.pad_value)

        scale_img_expanded = np.expand_dims(scale_img_padded, axis=0)

        heatmap, paf = predict_func(scale_img_expanded)

        heatmap = cv2.resize(heatmap[0], (0,0), fx=cfg.stride, fy=cfg.stride, interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:scale_img_padded.shape[0] - pad[2], :scale_img_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        paf = cv2.resize(paf[0], (0,0), fx=cfg.stride, fy=cfg.stride, interpolation=cv2.INTER_CUBIC)
        paf = paf[:scale_img_padded.shape[0] - pad[2], :scale_img_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (w, h), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    raw_heatmap_shown = np.maximum(0, heatmap_avg[:, :, 0:1] * 255)
    heatmap_shown = cv2.applyColorMap(raw_heatmap_shown.astype(np.uint8), cv2.COLORMAP_JET)
    img_with_heatmap = cv2.addWeighted(heatmap_shown, 0.5, img, 0.5, 0)

    cv2.imwrite('heatmap_shown.jpg', img_with_heatmap)


if __name__ == '__main__':

    img_id = 163640
    img_path = os.path.join('coco/val2017', '%012d.jpg' % img_id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data', default=img_path)
    args = parser.parse_args()

    predict(args)
