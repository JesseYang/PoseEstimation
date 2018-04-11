import functools
import matplotlib
from scipy.ndimage.filters import gaussian_filter
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

    all_peaks = []
    peak_counter = 0

    for part in range(cfg.ch_heats - 1):
        map_ori = heatmap_avg[:, :, part]
        map_flt = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map_flt.shape)
        map_left[1:,:] = map_flt[:-1,:]
        map_right = np.zeros(map_flt.shape)
        map_right[:-1,:] = map_flt[1:,:]
        map_up = np.zeros(map_flt.shape)
        map_up[:,1:] = map_flt[:,:-1]
        map_down = np.zeros(map_flt.shape)
        map_down[:,:-1] = map_flt[:,1:]

        peaks_binary = np.logical_and.reduce((map_flt>=map_left, map_flt>=map_right, map_flt>=map_up, map_flt>=map_down, map_flt>cfg.thre1))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks_with_score))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks_with_score)

    # visualize
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    
    canvas = np.copy(img) # B,G,R order
    
    for i in range(18):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
            cv2.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
    
    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    cv2.imwrite('part.jpg', to_plot)

if __name__ == '__main__':

    img_id = 163640
    img_path = os.path.join('coco/val2017', '%012d.jpg' % img_id)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data', default=img_path)
    args = parser.parse_args()

    predict(args)
