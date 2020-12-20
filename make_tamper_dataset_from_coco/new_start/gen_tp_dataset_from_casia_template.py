"""
created by haoran
time : 20201209
"""
from PIL import Image
import cv2 as cv
import os
import pandas as pd
import numpy as np
import random
import traceback
import warnings
import matplotlib.pylab as plt
import skimage.morphology as dilation
import image_crop
class GenTpFromTemplate:
    def __init__(self, template_dir=None, image_dir=None):
        self.template_dir = template_dir
        self.image_dir = image_dir
        self.template_list = os.listdir(template_dir)
        self.image_list = os.listdir(image_dir)
        self.tp_image_save_dir = '../TempWorkShop/src'
        self.tp_gt_save_dir = '../TempWorkShop/gt'

        if os.path.exists(self.tp_image_save_dir):
            pass
        else:
            print('create dict:',self.tp_image_save_dir)
            os.mkdir(self.tp_image_save_dir)

        if os.path.exists(self.tp_gt_save_dir):
            pass
        else:
            print('create dict:', self.tp_gt_save_dir)
            os.mkdir(self.tp_gt_save_dir)


    def gen_method_both_fix_size(self,width = 320, height = 320):
        """
        1. resize template to 320
        2. using crop 320 data to generate
        :return:
        """
        # 1 resize template
        template_dir = self.template_dir
        image_dir = self.image_dir
        template_list = self.template_list
        image_list = self.image_list

        for idx,item in enumerate(template_list):
            gt_name = item
            print('%d / %d'%(idx,len(template_list)))
            I = Image.open(os.path.join(template_dir,item))
            # deal with channel issues
            if len(I.split()) != 2:
                I = I.split()[0]
            else:
                pass
            I = I.resize((width, height), Image.ANTIALIAS)
            I = np.array(I,dtype='uint8')
            I = np.where(I>128,1,0)
            I = np.array(I, dtype='uint8')

            # random choose two images from fix size coco dataset
            gt = I.copy()
            for i in range(999):
                img_1_name = random.sample(image_list,1)[0]
                img_2_name = random.sample(image_list,1)[0]
                _ = open
                if img_1_name == img_2_name:
                    if i == 998:
                        traceback.print_exc()
                    else:
                        continue
                else:
                    img_1 = Image.open(os.path.join(image_dir, img_1_name))
                    img_2 = Image.open(os.path.join(image_dir, img_2_name))
                    if len(img_1.split())!=3 or len(img_2.split()) != 3:
                        continue
                    else:
                        break

            try:
                img_1 = np.array(img_1, dtype='uint8')
                img_2 = np.array(img_2, dtype='uint8')

                tp_img_1 = img_1.copy()
                tp_img_1[:,:,0] = I * img_1[:,:,0]
                tp_img_1[:,:,1] = I * img_1[:,:,1]
                tp_img_1[:,:,2] = I * img_1[:,:,2]

                I_reverse = np.where(I == 1, 0, 1)
                tp_img_2 = img_2.copy()

                tp_img_2[:,:,0] = I_reverse * img_2[:,:,0]
                tp_img_2[:,:,1] = I_reverse * img_2[:,:,1]
                tp_img_2[:,:,2] = I_reverse * img_2[:,:,2]
            except Exception as e:
                print(img_1_name)
                print(img_2_name)
                print(e)
            tp_img = tp_img_1 + tp_img_2
            # GenTpFromTemplate.__show_img(self, tp_img)


            # prepare to save
            tp_img = np.array(tp_img,dtype='uint8')
            double_edge_gt = GenTpFromTemplate.__mask_to_double_edge(self,gt)
            tp_gt = np.array(double_edge_gt, dtype='uint8')

            tp_img = Image.fromarray(tp_img)
            tp_gt = Image.fromarray(tp_gt)

            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0]+'_'+img_2_name.split('.')[0])+'.png')
            tp_img.save(os.path.join(self.tp_image_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.jpg')

            tp_gt.save(os.path.join(self.tp_gt_save_dir,
                                     gt_name.split('.')[0]+'_'+img_1_name.split('.')[0] + '_' + img_2_name.split('.')[0]) + '.bmp')


    def __mask_to_double_edge(self, orignal_mask):
            """
            :param orignal_mask: 输入的是 01 mask图
            :return: 255 100 50 mask 图
            """
            # print('We are in mask_to_outeedge function:')
            try:
                mask = orignal_mask
                # print('the shape of mask is :', mask.shape)
                selem = np.ones((3, 3))
                dst_8 = dilation.binary_dilation(mask, selem=selem)
                dst_8 = np.where(dst_8 == True, 1, 0)
                difference_8 = dst_8 - orignal_mask

                difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
                difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
                double_edge_candidate = difference_8_dilation + mask
                double_edge = np.where(double_edge_candidate == 2, 1, 0)
                ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                    mask == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
                ground_truth = np.where(ground_truth == 305, 255, ground_truth)
                ground_truth = np.array(ground_truth, dtype='uint8')
                return ground_truth

            except Exception as e:
                print(e)

    def __show_img(self,img):
        try:
            plt.figure('show_img')
            plt.imshow(img)
            plt.show()
        except Exception as e:
            print(e)

    def __path_check(self):
        """
        进行输入路径的检查
        :return:
        """

    def __prepare_template(self):

        pass
    def __choose_image(self):
        pass
    def __match_to_tamper(self):
        pass
    def analyse_template_size(self,template_dir):
        try:
            template_list = os.listdir(template_dir)
            row, col = [], []

            for idx,item in enumerate(template_list):
                print(idx)
                template_path = os.path.join(template_dir,item)
                _ = Image.open(template_path)
                _size = _.size
                row.append(_size[0])
                col.append(_size[1])


            data = {'row':row,'col':col}
            df = pd.DataFrame(data)
            writer = pd.ExcelWriter('../TempWorkShop/my.xlsx')
            df.to_excel(writer)
            writer.save()
            print(df)
        except Exception as e:
            print(e)

class GenTpFromCasiaTemplate(GenTpFromTemplate):
    def __init__(self,template_dir=None, image_dir=None):
        template_dir = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\casia2groundtruth-master\CASIA 2 Groundtruth'
        image_dir = 'D:\实验室\图像篡改检测\数据集\COCO_320_CROP6'
        super(GenTpFromCasiaTemplate, self).__init__(template_dir,image_dir)
        pass

if __name__ == '__main__':
    template_dir = 'D:\实验室\图像篡改检测\篡改检测公开数据\CASIA\casia2groundtruth-master\CASIA 2 Groundtruth'
    # GenTpFromCasiaTemplate().analyse_template_size(template_dir)

    # you only need to using this cmd to gen
    GenTpFromCasiaTemplate().gen_method_both_fix_size()