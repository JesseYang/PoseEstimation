import math


import matplotlib
import matplotlib.pyplot as plt
from numpy import ma
import cv2
import numpy as np
from numpy import ma
from cfgs.config import cfg


colors = cfg.colors
limb_seq = cfg.limb_seq
stickwidth = cfg.stickwidth

def _id_to_idx(peaks, part_id):
    for idx, I in enumerate(peaks):
        if I[3] == part_id:
            res = idx
    return res


def _colorize_np(flow):
    """从超分辨率弄过来的
    Hue: represents for direction
    Saturation: represents for magnitude
    Value: Keep 255
    """
    rgb_or_bgr = 'bgr'
    shape_length = len(flow.shape)
    assert rgb_or_bgr.lower() in ['rgb', 'bgr']
    assert shape_length in [3, 4]

    # following operations are based on (b, h, w, 2) ndarray
    if shape_length == 3:
        flow = np.expand_dims(flow, axis = 0)
    batch_size, img_h, img_w = flow.shape[:3]
    a = np.arctan2(-flow[:,:,:,1], -flow[:,:,:,0]) / np.pi
    h = (a + 1.0) / 2.0 * 255                       # (-1, 1) mapped to (0, 255)
    s = np.sum(flow ** 2, axis = 3) ** 0.5 * 10
    v = np.ones((batch_size, img_h, img_w)) * 255

    # build hsv image
    hsv = np.stack([h, s, v], axis = -1).astype(np.uint8)

    # hsv to rgb/bgr
    mapping = cv2.COLOR_HSV2RGB if rgb_or_bgr.lower() == 'rgb' else cv2.COLOR_HSV2BGR
    res = np.stack([cv2.cvtColor(i, mapping)] for i in hsv)

    # keep shape
    if shape_length == 3:
        res = np.squeeze(res)

    return res

def visualize_keypoints(img, keypoints):
    """可视化Keypoints
    
    # Arguments
        img: RGB image
        keypoints: list of persons, which is a list of (y, x, flag)

    # Returns
        canvas:
    """
    if np.array(keypoints).ndim == 2:
        keypoints = [keypoints]
    canvas = img.copy()

    for i in range(len(keypoints)):
        for j in range(18):
            canvas = canvas.copy()
            if keypoints[i][j][2] == 0: continue
            center = (int(keypoints[i][j][1]), int(keypoints[i][j][0]))
            cv2.circle(canvas,center, 4, colors[j], thickness=-1)

    canvas = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)

    return canvas


def visualize_heatmaps(img, heatmaps):
    """可视化单张heatmap

    # Arguments
        img:
        heatmap:

    # Returns
        canvas:

    """
    res = []
    for i in range(heatmaps.shape[-1]):
        gray = heatmaps[:,:,i]
        h, w = gray.shape
        gray = np.reshape(gray, (h, w, 1)) * 50
        #.reshape((427, 640, 1)) * 50
        gray = gray.astype(np.uint8)
        hsv = cv2.applyColorMap(gray, cv2.COLORMAP_HSV)
        canvas = img.copy()
        canvas = cv2.addWeighted(canvas, 0.4, hsv, 0.6, 0)
        res.append(canvas)
    return res


def visualize_pafs(img, pafs):
    """可视化单张PAF

    # Arguments
        img:
        paf:

    # Returns
        data:

    """
    oriImg = img
    U = pafs[:,:,0] * -1
    V = pafs[:,:,1]
    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
    M = np.zeros(U.shape, dtype='bool')
    M[U**2 + V**2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    
    plt.figure()
    plt.imshow(oriImg, alpha = .5)
    s = 5
    Q = plt.quiver(X[::s,::s], Y[::s,::s], V[::s,::s], U[::s,::s],
                scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    fig.canvas.draw()

    data = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def visualize_peaks(img, peaks):
    """可视化Keypoints
    
    # Arguments
        img: RGB image
        peaks: [[(y, x, confidence, id)] * n] * 18

    # Returns
        canvas:
    
    """
    if np.array(peaks).ndim == 2:
        peaks = [peaks]

    canvas = img.copy()

    for part_idx, peaks_each_part in enumerate(peaks):
        for peak_each_person in peaks_each_part:
            if len(peak_each_person) == 0: continue
            canvas = canvas.copy()
            center = (int(peak_each_person[1]), int(peak_each_person[0]))
            cv2.circle(canvas, center, 4, colors[part_idx], thickness=-1)

    canvas = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    return canvas


def visualize_connections(img, peaks, all_connections):
    """可视化Keypoints
    
    # Arguments
        img:
        peaks:
        all_connections:

    # Returns
        canvas:
    
    """
    res = []
    for limb_idx in range(19):
        part_idx_a, part_idx_b = limb_seq[limb_idx]
        canvas = visualize_peaks(img, [peaks[part_idx_a], peaks[part_idx_b]])

        for connection in all_connections[limb_idx]:
            cur_canvas = canvas.copy()
            peak_id_a, peak_id_b = list(map(int, connection[:2]))
            score, confidence_a, confidence_b = list(map(float, connection[2:]))

            peak_idx_a, peak_idx_b = _id_to_idx(peaks[part_idx_a], peak_id_a), _id_to_idx(peaks[part_idx_b], peak_id_b)
            peak_a, peak_b = peaks[part_idx_a][peak_idx_a], peaks[part_idx_b][peak_idx_b]

            peak_a_y, peak_a_x = peak_a[:2]
            peak_b_y, peak_b_x = peak_b[:2]
            length = ((peak_a_y - peak_b_y) ** 2 + (peak_a_x - peak_b_x) ** 2) ** 0.5
            angle = math.degrees(math.atan2(peak_b_y - peak_a_y, peak_b_x - peak_a_x))
            center = ((peak_a_x+peak_b_x)//2,(peak_a_y+peak_b_y)//2)
            polygon = cv2.ellipse2Poly(center, (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, [0, 0, 128])

            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        res.append(canvas)
    
    return res


    # canvas = visualize_peaks(img, peaks)

    # scores = []
    # for i in all_connections:
    #     for j in i:
    #         scores.append(float(j[2]))
    # min_ = min(scores)
    # scores = [i-min_ for i in scores]
    # max_ = max(scores)
    # scores = [int(i*255/max_) for i in scores]

    # # print(scores)
    
    # score_cnt = 0
    # for limb_idx in range(18):
    #     part_idx_a, part_idx_b = limb_seq[limb_idx]
    #     for connection in all_connections[limb_idx]:
    #         peak_id_a, peak_id_b = list(map(int, connection[:2]))
    #         score, confidence_a, confidence_b = list(map(float, connection[2:]))
            

    #         peak_idx_a, peak_idx_b = _id_to_idx(peaks[part_idx_a], peak_id_a), _id_to_idx(peaks[part_idx_b], peak_id_b)
    #         # print(score)

    #         cur_canvas = canvas.copy()
    #         peak_a, peak_b = peaks[part_idx_a][peak_idx_a], peaks[part_idx_b][peak_idx_b]

    #         peak_a_y, peak_a_x = peak_a[:2]
    #         peak_b_y, peak_b_x = peak_b[:2]
    #         length = ((peak_a_y - peak_b_y) ** 2 + (peak_a_x - peak_b_x) ** 2) ** 0.5
    #         angle = math.degrees(math.atan2(peak_b_y - peak_a_y, peak_b_x - peak_a_x))
    #         center = ((peak_a_x+peak_b_x)//2,(peak_a_y+peak_b_y)//2)
    #         polygon = cv2.ellipse2Poly(center, (int(length/2), stickwidth), int(angle), 0, 360, 1)
    #         cv2.fillConvexPoly(cur_canvas, polygon, [scores[score_cnt], 255, 255])
    #         score_cnt += 1
    #         canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)


    # return canvas

def visualize_matchsticks(img, peaks, persons):
    """可视化Keypoints
    
    # Arguments
        img:
        peaks:
        persons:

    # Returns
        canvas:
    
    """
    canvas = visualize_peaks(img, peaks)
    res = []

    for person in persons:
        cur_canvas = canvas.copy()
        for limb_idx in range(19):
            part_idx_a, part_idx_b = limb_seq[limb_idx]
            peak_id_a, peak_id_b = person[part_idx_a], person[part_idx_b]
            if peak_id_a == -1 or peak_id_b == -1: continue
            peak_idx_a, peak_idx_b = _id_to_idx(peaks[part_idx_a], peak_id_a), _id_to_idx(peaks[part_idx_b], peak_id_b)

            peak_a, peak_b = peaks[part_idx_a][peak_idx_a], peaks[part_idx_b][peak_idx_b]
            if peak_a[2] == 0 or peak_b[2] == 0: continue
            peak_a_y, peak_a_x = peak_a[:2]
            peak_b_y, peak_b_x = peak_b[:2]

            length = ((peak_a_y - peak_b_y) ** 2 + (peak_a_x - peak_b_x) ** 2) ** 0.5
            angle = math.degrees(math.atan2(peak_b_y - peak_a_y, peak_b_x - peak_a_x))
            center = ((peak_a_x+peak_b_x)//2,(peak_a_y+peak_b_y)//2)
            polygon = cv2.ellipse2Poly(center, (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[limb_idx])
        res.append(cur_canvas)
    
    return res
