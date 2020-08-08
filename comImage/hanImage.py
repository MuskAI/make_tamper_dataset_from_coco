# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2019/6/4 9:53
# @Author  : xhh
# @Desc    :  图片hash值，与汉明距离的计算
# @File    : hanImage.py
import cv2
import numpy as np
import numpy as np


def phash(path):
    # 加载并调整图片为32*32的灰度图片
    img = cv2.imread(path)
    img1 = cv2.resize(img, (32, 32),cv2.COLOR_RGB2GRAY)

    # 创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img1

    # DCT二维变换
    # 离散余弦变换，得到dct系数矩阵
    img_dct = cv2.dct(cv2.dct(vis0))
    img_dct.resize(8,8)
    # 把list变成一维list
    img_list = np.array().flatten(img_dct.tolist())
    # 计算均值
    img_mean = cv2.mean(img_list)
    avg_list = ['0' if i<img_mean else '1' for i in img_list]
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,64,4)])

# 计算汉明距离
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return
    count = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            count += 1
    return count


h1 = phash('./imageData/img1.png')
h2 = phash('./imageData/img2.jpg')
h3 = phash('./imageData/img3.png')
print("img1和img2：",hamming_distance(h1, h2))
print("img1和img3：",hamming_distance(h1, h3))
print("img2和img3：",hamming_distance(h2, h3))