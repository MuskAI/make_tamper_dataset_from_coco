"""
created by haoran
time:8-24
cod10k 类
"""
import os,sys
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import traceback
import skimage.morphology as dilation
import poisson_image_editing
class COD10K():
    def __init__(self,img_path,mask_path):
        self.img_path = img_path
        self.mask_path = mask_path

        # 阈值
        self.area_percent_threshold = 0.5
        self.bbox_threshold = 0.1

        pass
    def required_condition(self,area_percent,bbox):
        """
        :param area_percent:
        :param bbox:
        :return:
        """
        if area_percent > self.area_percent_threshold:
            print('面积超出阈值')
            return 'area_over_threshold'
        else:
            return 'area_ok'
        pass
    def gen_data_pair(self,img_path,mask_path):

        tamper_img_list = []
        mask_list = []
        for t_img in os.listdir(img_path):
            if 't' in t_img.replace('.tif',''):
                tamper_img_list.append(os.path.join(mask_path, t_img))
                mask_list.append(os.path.join(mask_path,t_img.replace('.tif','').replace('t','forged')+'.tif'))
            else:
                continue

        return tamper_img_list, mask_list
        pass
    def check_edge(self):
        pass
    def get_double_edge(self):
        mask = Image.open(self.mask_path)
        mask = np.array(mask)[:,:,0]
        mask = np.where(mask==255,1,0)

        print('the shape of mask is :', mask.shape)
        selem = np.ones((3,3))
        dst_8 = dilation.binary_dilation(mask, selem=selem)
        dst_8 = np.where(dst_8 == True,1,0)

        difference_8 = dst_8 - mask
        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)

        double_edge_candidate = difference_8_dilation + mask
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1,
                                                                                                           50,
                                                                                                           0)  # 所以内侧边缘就是100的灰度值
        return ground_truth
        pass
    def random_crop(self):
        pass
    def max_save_area_crop(self):
        pass

    def crop(self,img, target_shape=(320, 320)):
        img_shape = img.shape
        height = img_shape[0]
        width = img_shape[1]
        random_height_range = height - target_shape[0]
        random_width_range = width - target_shape[1]

        if random_width_range < 0 or random_height_range < 0:
            print('臣妾暂时还做不到!!!')
            traceback.print_exc()
            return 'error'

        random_height = np.random.randint(0, random_height_range)
        random_width = np.random.randint(0, random_width_range)

        return img[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]]

    def random_area_to_fix_background(self,background_path, mask_path, tamper_num=1, bk_shape=(320, 320)):
        """
           输入一张背景图和一张mask图，该mask图的size不应需要与背景图一致
           :param background:
           :param mask: mask任意一个前景图的mask
           :param tamper_num :篡改的数量，默认为1
           :return: 篡改好了的图像和一张GT 是list
           """
        # read image
        background = Image.open(background_path)
        mask = Image.open(mask_path)
        mask = np.where(mask[:,:,0]==255,1,0)
        # 找到mask 的矩形区域
        oringal_background = background.copy()
        a = mask
        a = np.where(a != 0)
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
        cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        # 计算物体所占区域面积
        object_area_percent = cut_mask.size / (bk_shape[0] * bk_shape[1])

        # 以左上角的点作为参考点，计算可以paste的区域
        background_shape = background.shape
        object_area_shape = cut_mask.shape
        paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
        print('the permit paste area is :', paste_area)
        row1 = np.random.randint(0, paste_area[0])
        col1 = np.random.randint(0, paste_area[1])

        # 在background上获取mask的区域
        temp_background = background.copy()
        random_area = False
        if random_area == True:
            cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
            cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
            cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask
        else:
            # cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
            cut_area = temp_background
            cut_area[:, :, 0] = cut_area[:, :, 0] * mask
            cut_area[:, :, 1] = cut_area[:, :, 1] * mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * mask

        for i in range(3):
            row2 = np.random.randint(0, paste_area[0])
            col2 = np.random.randint(0, paste_area[1])
            if abs(row1 - row2) + abs(col1 - col2) < 10:
                print('随机选到的区域太近，最好重新选择')
            else:
                break

        # 判断object和bg的大小是否符合要求
        if paste_area[0] < 5 or paste_area[1] < 5:
            print('提醒：允许的粘贴区域太小')
        if paste_area[0] < 1 or paste_area[1] < 1:
            print('无允许粘贴的区域')
            return False, False, False
        # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

        tamper_image = []
        tamper_mask = []
        tamper_gt = []
        tamper_poisson = []
        for times in range(tamper_num):
            bk_mask = np.zeros((background_shape[0], background_shape[1]), dtype='uint8')
            bk_area = np.zeros((background_shape[0], background_shape[1], 3), dtype='uint8')
            bk_mask[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1]] = cut_mask
            bk_area[row2:row2 + object_area_shape[0], col2:col2 + object_area_shape[1], :] = cut_area

            background[:, :, 0] = background[:, :, 0] * np.where(bk_mask == 1, 0, 1)
            background[:, :, 1] = background[:, :, 1] * np.where(bk_mask == 1, 0, 1)
            background[:, :, 2] = background[:, :, 2] * np.where(bk_mask == 1, 0, 1)
            background = background + bk_area

            tamper_image.append(background)
            tamper_mask.append(bk_mask)
        # 调用save_method保存

        for index, item in enumerate(tamper_image):

            mask = tamper_mask[index]
            print('the shape of mask is :', mask.shape)
            selem = np.ones((3, 3))
            dst_8 = dilation.binary_dilation(mask, selem=selem)
            dst_8 = np.where(dst_8 == True, 1, 0)
            difference_8 = dst_8 - mask

            difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
            difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
            double_edge_candidate = difference_8_dilation + tamper_mask[index]
            double_edge = np.where(double_edge_candidate == 2, 1, 0)
            ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
                tamper_mask[index] == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
            tamper_gt.append(ground_truth)

            try:
                mask = Image.fromarray(tamper_mask[index])
            except Exception as e:
                print('mask to Image error', e)
            # 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
            foreground = Image.fromarray(bk_area)

            background = Image.fromarray(oringal_background)
            mask = bk_mask
            try:
                poisson_foreground = cv.cvtColor(np.asarray(foreground.convert('RGB')), cv.COLOR_RGB2BGR)
                poisson_background = cv.cvtColor(np.asarray(background), cv.COLOR_RGB2BGR)
                poisson_mask = np.asarray(mask)
                poisson_mask = np.where(poisson_mask == 1, 255, 0)
                poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,
                                                                            poisson_mask)
                poisson_fusion_image = Image.fromarray(cv.cvtColor(poisson_fusion_image, cv.COLOR_BGR2RGB))

                #
                # plt.figure('123')
                # plt.imshow(poisson_fusion_image)
                # plt.show()
                tamper_poisson.append(poisson_fusion_image)
                # return tamper_image, tamper_poisson, tamper_gt

                # 保存

            except Exception as e:
                traceback.print_exc()


if __name__ == '__main__':
    img_path = ''
    mask_path = 'H:\COD10K_resize\\\GT'
    for mask in os.listdir(mask_path):
        gt_path = os.path.join(mask_path,mask)
        gt = COD10K(img_path,gt_path).get_double_edge()
        plt.figure(1)
        plt.imshow(gt)
        plt.show()