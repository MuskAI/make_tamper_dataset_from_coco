# -*- coding: utf-8 -*-
# !/usr/bin/env python
# @Time    : 2019/6/4 10:09
# @Author  : xhh
# @Desc    :  结构相似度量，计算图片之间的相似度
# @File    : ssimImage.py
# @Software: PyCharm
from skimage.measure import compare_ssim
# from scipy.misc import imread
import numpy as np
import cv2

# 读取图片
img1 = cv2.imread('./imageData/img1.png')
img2 = cv2.imread('./imageData/img2.jpg')
img3 = cv2.imread('./imageData/img3.png')
img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
img3 = np.resize(img3, (img1.shape[0], img1.shape[1], img1.shape[2]))
ssim1 = compare_ssim(img1, img2, multichannel=True)
ssim2 = compare_ssim(img1, img3, multichannel=True)
ssim3 = compare_ssim(img2, img3, multichannel=True)
print("img1和img2",ssim1)
print("img1和img3",ssim2)
print("img2和img3",ssim3)