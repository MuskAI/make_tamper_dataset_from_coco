"""
created by haoran
2020-7-23
1. 设计一个算法让随机到的图片size尽量相同来减小变形
2. 拿着object在background中寻找一块需要贴上去的区域
3.
"""
from pycocotools.coco import COCO
import numpy as np
import traceback



def my_load_background(size, size_threshold):
    """

    :param size: the foreground size （320,320）
    :param size_threshold: 一个阈值
    :return: an background which satisfied requirement
    """
    COCO.loadImgs()
    COCO.loadImgs()
