"""
created by haoran
time :2020-8-1 20:00
实现输入一张图片，将其crop到指定size的图片，这里的图片是指numpy  array
输入的是一张单通道有类别的图，其中50代表mask区域，100代表内边缘，255代表外边缘，0代表背景区域
save_percent :是按照crop后所占篡改区域比例的一个超参数，默认为1，即尽量都保留，对于无法全保留的采取最大程度保留的策略
crop_size :希望得到的大小
"""
import numpy as np
def crop_v1(mask, save_percent =1,crop_size=(320,320), ):
    """

    :param mask:
    :param save_percent:
    :param crop_size:
    :return:
    """

    store_bg_percent = 0.1
    # 获取篡改区域
    copy_mask = mask.copy()
    copy_mask = np.where(copy_mask==0,1,0)
    copy_mask = np.where(copy_mask==1,0,1)

    # 获取篡改区域的bbox，以及bbox的中心点
    a = mask
    a = np.where(a != 0 )
    bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]),np.max(a[1])
    center_loc = np.min(a[0])+(np.max(a[0]) - np.min(a[0]))//2, np.min(a[1]) + (np.max(a[1]) - np.min(a[1]))//2
    bbox_size = np.max(a[0]) - np.min(a[0]),np.max(a[1]) - np.min(a[1])
    print('center_loc ：',center_loc)

    # 首先判断bbox的size情况
    print('bbox_size:',bbox_size)
    print('crop_size:',crop_size)
    if bbox_size[0] < crop_size[0]-int(crop_size[0]*store_bg_percent) and bbox_size[1]<int(crop_size[1] * store_bg_percent):
        # 整个区域都保留下来


