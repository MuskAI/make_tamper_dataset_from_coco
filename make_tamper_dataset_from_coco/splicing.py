"""
created by haoran
time:2020-7-12
version:1.0

1. divide dataset to 2 gathers, gather A ,B
2. A will be tampered by the object of B
3. required A :
4. required B :as much background as possible,it means that the area of objects should be smaller
5. we set A:B = 2:8 since the average number of objects is 7.6
6. the first step is to find A and B, in this file we using F: findTamperedGather(),findObjectsGathter()
7. the core of this task is to fetch  object from B and paste it to A. using F: makeSplicing()
8. to get better results,we need some constraint: F： rgbConstraint() lightConstraint() areaConstraint()
9. after you pasted,using F: modfiyEdge() to optimize
"""
from pycocotools.coco import COCO
import numpy as np
import cv2 as cv
import os
import random
import matplotlib.pyplot as plt
import skimage.io as io
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class SplicingType:
    def __init__(self, dataDir = None,dataType = None, ann_info =None, image = None):
        self.imgIds = None
        dataDir = r'D:\实验室\图像篡改检测\数据集\COCO'
        dataType = 'val2017'
        val_info = r'D:\实验室\图像篡改检测\数据集\COCO\annotations\annotations_trainval2017\annotations\instances_val2017.json'
        val_image = r'D:\实验室\图像篡改检测\数据集\COCO\val2017'
        coco = COCO(val_info)  # 导入验证集
        self.dataDir = dataDir
        self.dataType = dataType
        self.ann_info = val_info
        self.image = val_image
        self.coco = coco
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

        # get all images containing given categories, select one at random
        catIds = coco.getCatIds(catNms=['person', 'dog'])
        print('catIds:')
        print(catIds)
        # imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]  # 获取全部图片信息
        imgIds = coco.getImgIds(catIds=catIds)
        # imgIds = coco.getImgIds(imgIds=[335328])
        print('imgIds:')
        print(imgIds)
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0] # list to dict
        print('img:')
        print(img)
        print('The length of this img list is:')
        print(len(imgIds))
        # I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
        # plt.axis('off')
        # plt.imshow(I)
        # plt.show()

        # plt.imshow(I)
        # plt.axis('off')
        # print(catIds)
        # annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        # anns = coco.loadAnns(annIds)
        # coco.showAnns(anns)
        # plt.show()

        self.imgIds = imgIds
        self.catIds = catIds
        pass


    def find_tampered_gather(self):
        print('The follow print is belong to find_tamperd_gather()：')
        coco = self.coco
        imgIds = self.imgIds
        catIds = self.catIds
        dataDir = self.dataDir
        dataType = self.dataType
        imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]  # 获取全部图片信息

        # divide it to tampered gather
        random.seed(133)
        tampered_gather = random.sample(imgs, int(len(imgs)*0.7))
        for x in tampered_gather:
            imgs.remove(x)

        objects_gather = imgs
        print('objects_gather:',objects_gather)
        annIds = coco.getAnnIds(imgIds=objects_gather[0][1]['id'], catIds=catIds, iscrowd=None)
        print('annIds is :')
        print(annIds)
        anns = coco.loadAnns(annIds)
        print('anns is :')
        print(anns[0]['segmentation'][0])








        pass

    def find_objects_gather(self):
        pass

    def get_all_data(self):

        pass
    def get_one_to_draw_mask(self):
        print('The follow print is belong to find_tamperd_gather()：')
        coco = self.coco
        imgIds = self.imgIds
        catIds = self.catIds
        dataDir = self.dataDir
        dataType = self.dataType
        for i in range(2000):
            img = coco.loadImgs(imgIds[i])[0]
            I = io.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
            shape = self.shape = I.shape
            print((dataDir, dataType, img['file_name']))

            edge_image = np.zeros(shape= self.shape).astype(np.float32)
            print(shape)
            plt.imshow(edge_image)
            plt.axis('off')


            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            # coco.showAnns(anns)
            SplicingType.gen_edge_image(self,anns)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig('./mask_image_result/mask_%d.png' %i,bbox_inches='tight', pad_inches = 0)
            # plt.show()

            img = cv.imread('./mask_image_result/mask_%d.png' %i,0)

            img_edge = cv.Canny(img,100,200)

            cv.imwrite('./edge_image_result/mask_%d.png' %i,img_edge)
            plt.close()
        pass

    def gen_edge_image(self,anns):
        """

        :param anns:
        :return:
        """
        if len(anns)==0:
            return 0
        if 'segmentation' in anns[0]:
            datasetType = 'instances'
        else:
            raise Exception('In gen_edge_image an error happend')

        if datasetType == 'instances':
            ax = plt.gca()
            # ax.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            edge =[]
            for ann in anns:
                # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0] # 随机颜色
                c = [1,1,1]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            # edge.append(poly)
                            color.append(c)
                break


            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            # p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.2)
            # p = SplicingType.threshold_demo(p)
            # return edge
            # ax.add_collection(p)


            # return p



    def get_tamper_area(self, anns):
        pass

    def threshold_demo(image):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  # 把输入图像灰度化

        # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
        print("threshold value %s" % ret)
        return binary
        # cv.namedWindow("binary0", cv.WINDOW_NORMAL)
        # cv.imshow("binary0", binary)



if __name__ == '__main__':
    # train_info = ''
    # train_image = ''
    # dataDir = r'D:\实验室\图像篡改检测\数据集\COCO'
    # dataType = 'val2017'
    # val_info = r'D:\实验室\图像篡改检测\数据集\COCO\annotations\annotations_trainval2017\annotations\instances_val2017.json'
    # val_image = r'D:\实验室\图像篡改检测\数据集\COCO\val2017'
    S = SplicingType().get_one_to_draw_mask()
