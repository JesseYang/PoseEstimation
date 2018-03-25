import math
from operator import itemgetter

import numpy as np
from scipy.ndimage.filters import gaussian_filter


from cfgs.config import cfg


limb_seq = cfg.limb_seq
# mapIdx = cfg.map_idx

def is_isolated(a, b):
    membership = ((a>=0).astype(int) + (b>=0).astype(int))
    return len(np.nonzero(membership == 2)[0]) == 0

def _id_to_idx(peaks, part_id):
    for idx, I in enumerate(peaks):
        if I[3] == part_id:
            res = idx
    
    return res


def get_peaks(heatmap, threshold):
    """从confidence maps中得到peaks, 热力图上每个peak表示了一个part

    # Arguments
        heatmap: 热力图，
        threshold: 小于threshold的peak将被忽略
    
    # Returns
        all_peaks: 每个元素: (y, x, confidence, id)
    """
    all_peaks = []
    peak_counter = 0

    # 数据集中人体共18个parts，对每一个part做遍历
    for part in range(18):
        # 得到该part对应的热力图(h, w)
        map_ori = heatmap[:, :, part]
        # 对热力图做平滑处理
        map = gaussian_filter(map_ori, sigma = 3)

        # 接下来的操作要遍历每一个像素，基于它的邻居搞事情
        # 为了GPU加速，通过平移的方法构造了邻居矩阵
        # 使得对于同一个位置x, map_left[x]是map[x]的左邻居，map_up[x]是map[x]的上邻居，等等
        # 从而使得元素间的操作变为矩阵操作
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        # 取得所有的peaks, 顾名思义，比周围像素都高的点，为了减少零星的噪声，引入了阈值
        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > threshold))
        # nonzero得indices, 这里换了axis
        peaks = list(zip(np.nonzero(peaks_binary)[0], np.nonzero(peaks_binary)[1]))

        # 每一个元素: (y, x, confidence)
        peaks_with_score = [x + (map_ori[x[0], x[1]],) for x in peaks]
        # 为每个peaks取独立的ID
        id = range(peak_counter, peak_counter + len(peaks))
        # 每个元素: (y, x, confidence, id)
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    
    return all_peaks


def get_connections(peaks, pafs, threshold):
    mid_num = 10
    all_connections, special_limb_idxs = [], []
    img_h = pafs.shape[0]

    for limb_idx in range(19):
        part_idx_a, part_idx_b = limb_seq[limb_idx]
        # 取两part的点以及对应limb的PAF
        candidates_a, candidates_b = peaks[part_idx_a], peaks[part_idx_b]
        paf = pafs[:,:,2*limb_idx:2*limb_idx+2]
        
        num_of_candidates_a, num_of_candidates_b = len(candidates_a), len(candidates_b)
        
        # 如果limb的一端不存在, 记录到special_limb_idxs中
        if num_of_candidates_a * num_of_candidates_b == 0:
            special_limb_idxs.append(limb_idx)
            all_connections.append([])
            continue
        
        # 收集connection_candidates
        connection_candidates = []
        for peak_a in candidates_a:
            for peak_b in candidates_b:
                peak_confidence_a, peak_confidence_b = peak_a[2], peak_b[2]
                peak_id_a, peak_id_b = peak_a[3], peak_b[3]
                # 求candidates_a到candidates_b的limb向量
                # limb_vec = np.subtract(candidate_a[:2], candidate_b[:2])
                peak_a_y, peak_a_x = peak_a[:2]
                peak_b_y, peak_b_x = peak_b[:2]
                limb_vec = np.array([peak_b_y - peak_a_y, peak_b_x - peak_a_x])
                length = ((peak_a_y - peak_b_y) ** 2 + (peak_a_x - peak_b_x) ** 2) ** 0.5
                # 求模长
                norm = np.linalg.norm(limb_vec)
                # 对于重合的part, 也跳过
                if length == 0: continue
                limb_vec = limb_vec / length

                # 论文提到的line integral是连续的, 实践中, 通过采样求和多个平均分布的点来做
                
                # 制作平均分布的点
                # 通过np.linspace可以得到均匀分布的矩阵
                # 如 np.linspace(0, 9, num=10)
                # >>> array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])
                startend = list(zip(np.linspace(peak_a_y, peak_b_y, num = mid_num), \
                                    np.linspace(peak_a_x, peak_b_x, num = mid_num)))
                vec_y = np.array([paf[int(y), int(x), 0] for y,x in startend])
                vec_x = np.array([paf[int(y), int(x), 1] for y,x in startend])
                # print(vec_y)
                
                # 求得分数
                score_midpts = np.multiply(vec_y, limb_vec[0]) + np.multiply(vec_x, limb_vec[1])
                score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * img_h / norm - 1, 0)
                #
                criterion1 = len(np.nonzero(score_midpts > threshold)[0]) > 0.8 * len(score_midpts)
                #
                criterion2 = score_with_dist_prior > 0
                # 分别是id_a, id_b, score, score + confidences
                # peak: y, x, confidence, id
                if criterion1 and criterion2:
                    connection_candidate = [peak_id_a, peak_id_b, score_with_dist_prior,  score_with_dist_prior + peak_confidence_a, peak_confidence_b]
                    connection_candidates.append(connection_candidate)
        
        # 优先取score更高的limb组合, 之前取id的原因在于sort后connection和peak的信息不匹配
        connection_candidates.sort(key = itemgetter(2), reverse = True)

        # axis 1分别为: peak_id_a, peak_id_b, 距离得分, confidence_a, confidence_b
        connections = np.zeros((0, 5))
        for connection_candidate in connection_candidates:
            peak_id_a, peak_id_b, 距离得分 = connection_candidate[:3]
            # print(peak_id_a, peak_id_b, 距离得分)
            peak_idx_a, peak_idx_b = _id_to_idx(peaks[part_idx_a], peak_id_a), _id_to_idx(peaks[part_idx_b], peak_id_b)

            confidence_a, confidence_b = peaks[part_idx_a][peak_idx_a][2], peaks[part_idx_b][peak_idx_b][2]
            # print(connections[:, 3])
            # print(connections[:, 4])
            if peak_id_a not in connections[:, 0] and peak_id_b not in connections[:, 1]:
                connections = np.vstack([connections, [peak_id_a, peak_id_b, 距离得分, confidence_a, confidence_b]])
                # 优先连接得分高的, 连够了break
                if len(connections) >= min(num_of_candidates_a, num_of_candidates_b): break
        # print(num_of_candidates_a, num_of_candidates_b, len(connections))
        all_connections.append(connections)
    
    return all_connections, special_limb_idxs


def multi_person_parse(peaks, all_connections, special_limb_idx):
    def find(persons, idx_a, idx_b, id_a, id_b):
        """
        idx_a, idx_b: limb两端的part_idx
        id_a, id_b: 查找的part_id
        
        # Returns
            res: 这两块肉出现在哪些人身上(idx)
        """
        res = []
        for idx, person in enumerate(persons):
            if person[idx_a] == id_a or person[idx_b] == id_b:
                res.append(idx)
                
        return res
    # person:
    # 0~17位 - 对应每个part的part_idx(就是从哪个地方取的……)
    # 18位 - part的个数
    # 19位 - 得分()
    persons = -1 * np.ones((0, 20))

    for limb_idx in range(19):
        if limb_idx in special_limb_idx: continue
        part_idx_a, part_idx_b = limb_seq[limb_idx]
        for connection in all_connections[limb_idx]:
            peak_id_a, peak_id_b = list(map(int, connection[:2]))
            peak_idx_a, peak_idx_b = _id_to_idx(peaks[part_idx_a], peak_id_a), _id_to_idx(peaks[part_idx_b], peak_id_b)
            # 在已分配的人中找这两块肉(id_a, id_b)的分配情况
            found = find(persons, part_idx_a, part_idx_b, peak_id_a, peak_id_b)

            isolated = is_isolated(persons[found[0]][:-2], persons[found[1]][:-2]) if len(found) == 2 else None

            # 未分配, 新建一个person
            if len(found) == 0 and limb_idx < 17:
                person = -1 * np.ones(20)
                # 安装部件
                person[[part_idx_a, part_idx_b]] = peak_id_a, peak_id_b
                # 已拼装的部件数为2
                person[-1] = 2
                # 总得分: 边权 + 点权
                person[-2] = peaks[part_idx_a][peak_idx_a][2] + peaks[part_idx_b][peak_idx_b][2] + connection[2]
                # 相当于append
                persons = np.vstack([persons, person])
        
            # 两块part只分配了一块
            # 满足了两次person[idx_a] == id_a条件
            # 比如脖子a, 在脖子->左肩的匹配中放到了甲身上, 在脖子->右肩的匹配中被放到了乙身上
            elif (len(found) == 1 and persons[found[0]][part_idx_b] != peak_id_b) or\
            (len(found) == 2 and not isolated):
                persons[found[0]][part_idx_b] = peak_id_b
                persons[found[0]][-1] += 1
                persons[found[0]][-2] += peaks[part_idx_b][peak_idx_b][2] + connection[2]

            # 满足了一次person[idx_a] == id_a条件，一次person[idx_b] == id_b条件
            # 两个人各缺一半, 拼成一个人
            elif len(found) == 2 and isolated:
                persons[found[0]][:-2] = (persons[found[0]][:-2] + 1) + persons[found[1]][:-2]
                persons[found[0]][-2:] += persons[found[1]][-2:]
                persons[found[0]][-2] += connection[2]
                persons = np.delete(persons, found[1], 0)
            
            if limb_idx == 18:
                print(connection, len(found))
    
    # delete some rows of subset which has few parts occur
    del_idxs = []
    for i in range(len(persons)):
        if persons[i][-1] < 4 or persons[i][-2] / persons[i][-1] < 0.4:
            del_idxs.append(i)
    persons = np.delete(persons, del_idxs, axis = 0)
    
# 后处理也并不完美, 在get-started/predict.ipynb中用gt_heatmap, gt_paf跑出来的单个人被分成了三个部分
# 原因在于这里的"isolated"条件, 只要两个被组装的"人"有重合的part就不会被组装

    return persons
