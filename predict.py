import functools
from pathlib import Path


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

def load_labels(path):
    pass


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



    # img_h, img_w = raw_img.shape[:2]
    # img = np.expand_dims(raw_img, 0)
    # heatmap_avg = np.zeros((img_h, img_w, cfg.ch_heats))
    # paf_avg = np.zeros((img_h, img_w, cfg.ch_vectors))

    # heatmap, paf = pred_func(img)
    # b = np.squeeze(heatmap[:,:,0])
    # print(b.shape)
    # b = cv2.resize(b, None, fx = 8, fy = 8, interpolation=cv2.INTER_CUBIC)
    # print(b.shape)
    # quit()

    # resized_heatmaps = np.array([cv2.resize(np.squeeze(heatmap[:,:,i]), (0,0), 8, 8) for i in range(19)])

    # print(resized_heatmaps.shape)
    # quit()
    # # heatmap = cv2.resize(heatmap, None, 8, 8)
    # print(heatmap.shape)
    # quit()
    # outputs = visualize_heatmaps(raw_img, heatmap)
    # for idx, output in enumerate(outputs):
    #     imageio.imsave('heatmap-{}.png'.format(idx), output)
    # print(np.min(heatmap), np.max(heatmap))
    # print(np.min(paf), np.max(paf))

    # quit()

    # peaks = get_peaks(heatmap_avg, cfg.th1)
    # all_connections, special_limb_idx = get_connections(peaks, paf_avg, cfg.th2)
    # persons = multi_person_parse(peaks, all_connections, special_limb_idx)
    
    # res = visualize_matchsticks(raw_img, peaks, persons)


    # heatmap_outputs = visualize_heatmaps(raw_img, heatmap)
    # # print(type(res), res.shape)
    # print(len(peaks), len(all_connections), len(persons))
    # print(len(res))
    # for idx, res_ in enumerate(res):
    #     print(res_.shape)
    #     imageio.imsave('output{}.png'.format(idx), res_)
    # return res


def _eval(img, labels, predict_func):
    pass


def predict(args):
    # prepare predictor
    sess_init = SaverRestore(args.model_path)
    model = Model('test')
    predict_config = PredictConfig(session_init = sess_init,
                                   model = model,
                                   input_names = ['imgs'],
                                   output_names = ['HeatMaps', 'PAFs'])
    predict_func = OfflinePredictor(predict_config)

    # load data
    # get_data(txt_path)
    # if args.from_sintel is None:
    #     inputs = load_frames(args.input_path)
    # else:
    #     inputs = sintel_helper(args.input_path)

    img = imageio.imread(args.input_path)


    # if args.eval:
    heatmap, paf = _predict(img, predict_func)
    # else:
    #     print('ww')
    #     labels = load_labels(args.label_path)
    #     _eval(img, labels, predict_func)

    print(heatmap, paf)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of model', required = True)
    parser.add_argument('--input_path', help='path of input data', required = True)
    parser.add_argument('--label_path', help='path of input data')
    parser.add_argument('--eval', help='path of input data', action='store_true')
    parser.add_argument('--output_path', help='path of outputs')
    args = parser.parse_args()

    predict(args)
