"""
created by haoran
time 2020/12/18
tips:
1. 通过纹理数据来生成篡改数据


"""
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import os,sys
import warnings
import traceback
import skimage.morphology as dilation

class TamperDataset:
    def __init__(self, src_dir, mask_dir, workshop_dir):
        """
        拥有原始数据src, 对应的mask图
        :param src_dir:
        :param mask_dir:
        :param workshop_dir:
        """
        # 0 变量
        self.src_dir = src_dir
        self.mask_dir = mask_dir
        self.workshop_dir = workshop_dir

        self.data320 = os.path.join(workshop_dir,'crop_or_resize_320')
        self.data320_src = os.path.join(self.data320,'src')
        self.data320_mask = os.path.join(self.data320,'gt')


        self.img_tamper_save_path = os.path.join(workshop_dir,'src')
        self.img_gt_save_path = os.path.join(workshop_dir,'gt')
        ###############################
        self.bk_shape = (320,320)
        self.tamper_num = 1
        # 1 path check
        if not self.path_check():
            NotADirectoryError('请检查输入路径')
            exit(1)

        # 2 把图片处理成320并保存在相应目录下
        self.__data_prepare()

        # 3 生成数据



    def path_check(self):
        if not os.path.exists(self.src_dir):
            return False

        if not os.path.exists(self.mask_dir):
            return False

        if not os.path.exists(self.workshop_dir):
            os.mkdir(self.workshop_dir)
            print('created dir:',self.workshop_dir)

        return True

    def __data_prepare(self):
        """
        不管是cm 还是sp类型都需要通过这个函数来进行处理， 使不同size的图片满足输入320的要求
        :return:320 src and gt ， numpy
        """
        src_list = os.listdir(self.src_dir)
        mask_list = os.listdir(self.mask_dir)

        # 1 deal with size problem
        # if the size of image >=320, just crop it
        # if the size of image <320,  we need to resize it
        # but we can't resize gt directly
        # read the image and deal with it one by one

        for idx,item in enumerate(src_list):
            src_path = os.path.join(self.src_dir,item)
            if item == mask_list[idx]:
                gt_path = os.path.join(self.mask_dir, mask_list[idx])
            else:
                traceback.print_exc('when match src and gt rise an error')
                exit(1)

            src = Image.open(src_path)
            gt = Image.open(gt_path)

            src_size = src.size
            gt_size = gt.size

            # 再次确认两张图是否匹配
            if src_size == gt_size:
                pass
            else:
                traceback.print_exc('when match src and gt rise an error')
                exit(1)


            if src_size[0]>=320 or src_size[1]>=320:
                # 第一种情况>=320 需要进行crop
                # crop 返回的是裁剪好的numpy 数组 和一个tuple(height,width)
                _flag_src, _pos_src = self.crop(np.array(src,dtype='uint8'))
                _flag_gt, _pos_gt = self.crop(np.array(gt,dtype='uint8'),pos=_pos_src)
                if _flag_src == 'error' or _flag_gt == 'error':
                    continue
                else:
                    src_crop_or_resize = _flag_src
                    gt_crop_or_resize = _flag_gt


                #########################################
            else:
                # 第二种情况<320 需要进行resize
                # 都resize,这里的mask 是 0 255
                src_crop_or_resize = src.resize((320, 320), Image.ANTIALIAS)
                src_crop_or_resize = np.array(src_crop_or_resize,dtype='uint8')
                gt_crop_or_resize = gt.resize((320, 320), Image.ANTIALIAS)
                gt_crop_or_resize = np.array(gt_crop_or_resize,dtype='uint8')
                gt_crop_or_resize = np.where(gt_crop_or_resize>150, 255,0)


            # 现在size 都处理好了，下面要进行的任务就是tamper，结果都是numpy数组
            try:
                src_crop_or_resize = Image.fromarray(src_crop_or_resize)
                gt_crop_or_resize = Image.fromarray(gt_crop_or_resize)
                src_crop_or_resize.save(os.path.join(self.data320_src,item.split('.')[0]+'_%d_%d.png' % _pos_src))
                gt_crop_or_resize.save(os.path.join(self.data320_mask,item.split('.')[0]+'_%d_%d.png' % _pos_src))

            except Exception as e:
                traceback.print_exc(e)



    def splicing(self):
        pass
    def copy_move(self,img_path,mask_path):
        """
        对输入的img 和 mask 进行copy move 粘贴，简单来说就是对同一张图上物体平移或者旋转一定的位置

        :param img_path:是一张图片而不是一个文件夹
        :param mask_path:同上
        :return: 直接保存图片不返回
        """
        # read image
        background = Image.open(img_path)
        background = np.array(background)
        mask = Image.open(mask_path).convert('RGB')
        mask = np.array(mask)
        mask = np.where(mask[:,:,0]==255,1,0)
        ##########################################

        # 找到mask 的矩形区域
        oringal_background = background.copy()
        a = mask
        a = np.where(a != 0)
        bbox = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
        cut_mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        cut_area = oringal_background[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        #########################################


        # 计算物体所占区域面积
        object_area_percent = cut_mask.size / (self.bk_shape[0] * self.bk_shape[1])

        # 以左上角的点作为参考点，计算可以paste的区域
        background_shape = background.shape
        object_area_shape = cut_mask.shape
        paste_area = [background_shape[0] - object_area_shape[0], background_shape[1] - object_area_shape[1]]
        print('the permit paste area is :', paste_area)
        row1 = np.random.randint(0, paste_area[0])
        col1 = np.random.randint(0, paste_area[1])

        # 在background上获取mask的区域
        temp_background = background.copy()
        random_area = False # 这里的random_area 是否只扣取mask 所在的区域， 默认为是的 so random_area = False
        if random_area == True:
            cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
            cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
            cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask
        else:
            # cut_area = temp_background[row1:row1 + object_area_shape[0], col1:col1 + object_area_shape[1], :]
            cut_area[:, :, 0] = cut_area[:, :, 0] * cut_mask
            cut_area[:, :, 1] = cut_area[:, :, 1] * cut_mask
            cut_area[:, :, 2] = cut_area[:, :, 2] * cut_mask
            # plt.figure(1)
            # plt.imshow(cut_area)
            # plt.show()

        for i in range(5):
            row2 = np.random.randint(0, paste_area[0])
            col2 = np.random.randint(0, paste_area[1])
            if abs(row1 - row2) + abs(col1 - col2) < 50:
                print('随机选到的区域太近，最好重新选择')
            else:
                break

        # # 判断object和bg的大小是否符合要求
        # if paste_area[0] < 5 or paste_area[1] < 5:
        #     print('提醒：允许的粘贴区域太小')
        # if paste_area[0] < 1 or paste_area[1] < 1:
        #     print('无允许粘贴的区域')
        #     return False, False, False
        # 随机在background上贴上该mask的区域，并且保证与原区域有一定的像素偏移,然后生成新的mask图

        tamper_image = []
        tamper_mask = []
        tamper_gt = []

        for times in range(self.tamper_num):
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

        # 下面是生成GT的过程
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

        ############################################################################

            # 下面是准备保存的代码
            try:
                # 保存
                for index,t_img in enumerate(tamper_image):
                    t_img = Image.fromarray(t_img)
                    t_img.save(self.img_tamper_save_path)

                for index, t_gt in enumerate(tamper_gt):
                    t_img = Image.fromarray(t_gt).convert('RGB')
                    t_img.save(self.img_gt_save_path)

            except Exception as e:
                traceback.print_exc(e)
            ##########################################
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
    def crop(self,img, target_shape=(320, 320),pos = None):
        """
        :param img: numpy array
        :param target_shape:
        :return:
        """
        if pos == None:
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
        else:
            random_height = pos[0]
            random_width = pos[1]

        return img[random_height:random_height + target_shape[0], random_width:random_width + target_shape[1]],\
               (random_height,random_width)


class TextureTamperDataset(TamperDataset):
    def __init__(self,src_dir, mask_dir, workshop_dir):
        super(TextureTamperDataset, self).__init__(src_dir, mask_dir, workshop_dir)
        self.edge_dir = ''
        pass



