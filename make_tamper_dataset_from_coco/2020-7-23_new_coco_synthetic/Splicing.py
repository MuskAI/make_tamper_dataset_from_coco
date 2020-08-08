"""
created by haoran
2020-7-23
一系列的函数来实现篡改
"""
from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import pylab
import os
import sys
import time
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import pdb
from get_double_edge import mask_to_outeedge
from image_Fusion import Possion
import poisson_image_editing
import skimage.morphology as dilation
import traceback
matplotlib.use('Qt5Agg')
# def parse_args():
#   """
#   Parse input arguments
#   """
#   parser = argparse.ArgumentParser(description='input begin and end category')
#   parser.add_argument('--begin', dest='begin',
#             help='begin type of cat', default=None, type=int)
#   parser.add_argument('--end', dest='end',
#             help='begin type of cat',
#             default=None, type=int)
#
#   if len(sys.argv) == 1:
#     parser.print_help()
#     sys.exit(1)
#
#   args = parser.parse_args()
#   return args
#

# args=parse_args()
# print(args.begin)
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
dataDir='D:\\实验室\\图像篡改检测\\数据集\\COCO\\'
dataType='val2017'
annFile='%s/annotations/instances_%s.json'%(dataDir,dataType)
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
for cat in cats[2:3]:
    for num in range(2000):
        try:
            catIds = coco.getCatIds(catNms=[cat['name']])
            imgIds = coco.getImgIds(catIds=catIds )
            img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            I=io.imread(os.path.join(dataDir,dataType,'{:012d}.jpg'.format(img['id'])))

            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            bbx=anns[0]['bbox']
            mask=np.array(coco.annToMask(anns[0]))
            print(np.shape(mask))
            print(np.shape(I))

            I1=I
            # mask 是 01 蒙版
            I1[:,:,0]=np.array(I[:,:,0] * mask )
            I1[:,:,1]=np.array(I[:,:,1] * mask )
            I1[:,:,2]=np.array(I[:,:,2] * mask )
            # differece_8是background的edge
            difference_8 = mask_to_outeedge(mask)

            # check the outer edge is right
            # feedback1: generate difference_8 wrong
            check_outer_edge = np.where(difference_8 == 1,0.5,0)+mask
            # plt.figure('check_outer_edge')
            # plt.imshow(check_outer_edge)
            # plt.show()

            difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3,3)))
            difference_8_dilation = np.where(difference_8_dilation == True,1,0)
            # double_edge_candidate = difference_8_dilation - difference_8
            double_edge_candidate = difference_8_dilation + mask
            double_edge = np.where(double_edge_candidate == 2, 1, 0) # here we get double edge,now to confirm it
            confirm_image = np.where(double_edge==1,0.5,0) + np.where(difference_8 == 1,0.8,0)
            ground_truth = np.where(double_edge==1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1, 50 , 0) # 所以内侧边缘就是100的灰度值
            # cv2.imwrite('../ground_truth/double_edge_ground_truth_%d.jpg' %num, ground_truth)

            plt.imsave('./confirm/double_edge_confirm_image_%d.png' %num, confirm_image)

            # plt.imsave('double_edge_image.png',double_edge)

            rand=np.random.randint(100,size=1)[0]
            #flag=0
                #I1=cv2.GaussianBlur(I1,(5,5),0)
                #flag=1
            img1 = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            cv2.imwrite('../ground_truth/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(bbx[0]) + '_' + str(
                        bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' + cat['name'] + '.tif',ground_truth)
            b1=io.imread(os.path.join(dataDir,dataType,'{:012d}.jpg'.format(img1['id'])))
            text_img = Image.new('RGBA', (np.shape(b1)[0],np.shape(b1)[1]), (0, 0, 0, 0))
            background=Image.fromarray(b1,'RGB')
            foreground=Image.fromarray(I1,'RGB').convert('RGBA')
            datas=foreground.getdata()

            newData = []
            for item in datas:
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    newData.append((0, 0, 0, 0))
                else:
                    newData.append(item)
            foreground.putdata(newData)
            foreground=foreground.resize((background.size[0],background.size[1]),Image.ANTIALIAS) # 抗锯齿


            try:
                mask = Image.fromarray(mask)
            except Exception as e:
                print('mask to Image error', e)

            mask = mask.resize((background.size[0],background.size[1]),Image.ANTIALIAS)

            print('mask size is :',mask.size)
            print('')
            # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
            try:
                poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
                poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
                poisson_mask =np.asarray(mask)
                poisson_mask = np.where(poisson_mask == 1,255,0)
                poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background, poisson_mask)
                poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))
            except Exception as e:
                traceback.print_exc()

            background.paste(foreground,(0,0),mask=foreground.split()[3])
            # if rand%3<2:
            # 	background=background.filter(ImageFilter.GaussianBlur(radius=1.5))
            if not os.path.isfile('../filter_tamper2/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(bbx[0])+'_'+str(bbx[1])+'_'+str(bbx[0]+bbx[2])+'_'+str(bbx[1]+bbx[3])+'_'+cat['name']+'.png'):
                print('../filter_tamper2/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(bbx[0])+'_'+str(bbx[1])+'_'+str(bbx[0]+bbx[2])+'_'+str(bbx[1]+bbx[3])+'_'+cat['name']+'.png')
                print(background)
                # background = background.astype(np.uint8)
                # cv2.imwrite('../filter_tamper/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(bbx[0])+'_'+str(bbx[1])+'_'+str(bbx[0]+bbx[2])+'_'+str(bbx[1]+bbx[3])+'_'+cat['name']+'.png',background)
                # io.imsave('../filter_tamper/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(bbx[0])+'_'+str(bbx[1])+'_'+str(bbx[0]+bbx[2])+'_'+str(bbx[1]+bbx[3])+'_'+cat['name']+'.png',background)
                background.save('../filter_tamper2/Tp_'+str(img['id'])+'_'+str(img1['id'])+'_'+str(bbx[0])+'_'+str(bbx[1])+'_'+str(bbx[0]+bbx[2])+'_'+str(bbx[1]+bbx[3])+'_'+cat['name']+'.png')
                poisson_fusion_image.save(
                    '../filter_tamper2_poisson_fusion/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(bbx[0]) + '_' + str(
                        bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' + cat['name'] + '.png')
        except Exception as e:
            print(e)
print('finished')



def