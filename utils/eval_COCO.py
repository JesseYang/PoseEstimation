import sys
sys.path.insert(1, '../data/coco/PythonAPI/')
from pycocotools.coco import COCO
import numpy as np
from cfgs.config import cfg

ours_to_coco = cfg.ours_to_coco

def evaluate(output_dir):
    
    pass




def outputs_to_json(outputs, json_name):
    """转换outputs格式, 存成可以用COCO-analyze分析的json文件
    """
    pass


coco_to_ours = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
ours_to_coco = cfg.ours_to_coco

a = np.arange(18)
b = a[coco_to_ours][ours_to_coco]
print(b == a)
