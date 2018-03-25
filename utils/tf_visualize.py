import tensorflow as tf
from .visualize import (
    visualize_keypoints as np_visualize_keypoints,
    visualize_peaks as np_visualize_peaks,
    visualize_matchsticks as np_visualize_matchsticks,
    visualize_heatmap as np_visualize_heatmap,
    visualize_paf as np_visualize_paf
)

def visualize_keypoints(img, keypoints):
    return tf.py_func(np_visualize_keypoints, [img, keypoints], tf.float32)


cv2.circle(canvas,center, 4, colors[j], thickness=-1)



tf_painter



def circle(img, center, radius, color, thickness = 1, lineType = 8, shift = 0):
    pass