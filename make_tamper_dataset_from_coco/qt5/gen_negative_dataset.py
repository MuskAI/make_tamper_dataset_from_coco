"""
Created by haoran
time: 10/12
description:Using this file to gen negative dataset from random image dataset ,default:COCO
"""
import os,sys
import cv2 as cv
from PIL import Image
import traceback
import random
import numpy as np
import shutil

class NegativeDataset():
    def __init__(self):
        pass
    def coco(self, in_path = None, save_path = None, number = 0, target_shape=(320, 320)):
        """
        从COCO数据集生成篡改数据
        :param path:存放图片的文件夹路径
        :return:
        """
        # 0 判断输入的合法性
        if in_path == None:
            print('You should giving a useful COCO path')
            traceback.print_exc()
        else:
            pass

        if number == 0:
            print('In test phase using default number:10')
            number = 10
        else:
            pass

        if os.path.exists(save_path):
            print('路径存在，检查是否为空文件夹')
            if len(os.listdir(save_path)) != 0:
                print('保存路径不为空，请重新输入')
                sys.exit()
            else:
                pass
        else:
            print('保存路径不存在，开始创建')
            os.mkdir(save_path)

        # 1 打开路径，获取图片信息,划分图片
        coco_file_path = in_path
        image_list = os.listdir(coco_file_path)
        if number <= len(image_list):
            pass
        else:
            print('the number is too big')
            traceback.print_exc()
        image_list = random.sample(image_list, number)
        for index,img in enumerate(image_list):
            t_img_path = os.path.join(coco_file_path,img)
            t_img = Image.open(t_img_path)
            t_img = np.array(t_img)
            t_img_crop = NegativeDataset.__crop(self, t_img, target_shape)
            # print('the ',type(t_img_crop))
            # if type(t_img_crop) == type(None):
            #     print(img,'error')
            #     continue
            if t_img_crop.shape[:-1] != target_shape:
                continue

            t_img_crop = Image.fromarray(t_img_crop)
            t_img_crop.save(os.path.join(save_path, img))
            print('\r', 'Saving process: %d/%d'%(index, len(image_list)), end='')

    def __crop(self, img, target_shape=(320, 320)):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        random_height_range = height - target_shape[0]
        random_width_range = width - target_shape[1]

        if random_width_range < 1 or random_height_range < 1:
            return img
        random_height = np.random.randint(0, random_height_range)
        random_width = np.random.randint(0, random_width_range)

        return img[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]]

    def gt(self,target_shape=(320, 320)):
        """
        调用这个方法，返回指定尺寸大小的纯黑numpy数组
        :return:
        """
        return np.zeros([target_shape[0],target_shape[1]],dtype='uint8')
    def rename(self,path):
        for index, item in enumerate(os.listdir(path)):
            os.rename(os.path.join(path,item),os.path.join(path,'negative_'+item))
            print('\r', 'The rename process: %d/%d'%(index, len(os.listdir(path))),end='')
if __name__ == '__main__':
    # NegativeDataset().coco(in_path='D:\\实验室\\图像篡改检测\\数据集\\COCO\\train2017',
    #                        save_path='D:\\实验室\\图像篡改检测\\数据集\\COCO_320_CROP',number=10000)
    NegativeDataset().rename(r'H:\10月数据准备\10月12日实验数据\negative\COCO_320_CROP')