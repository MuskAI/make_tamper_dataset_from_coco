import os
import numpy as np
from PIL import Image, ImageFilter
from image_squeene import compress_image, get_size, MyGaussianBlur
import random
from sklearn.model_selection import train_test_split
from PIL import ImageFile
import traceback
import cv2
import sys
from gen_8_map import gen_8_2_map as gen_8_map

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataParser():
    def __init__(self, batch_size_train):
        self.train_file = 'C:\\Users\\musk\\Desktop\\test_cp\\tamper_result'
        self.double_edge_file = 'C:\\Users\\musk\\Desktop\\test_cp\\ground_truth_result'
        self.train_file = '/home/liu/chenhaoran/datasets/tamper_result'
        # self.doublecxckerk===scsc_edge_file = '/home/liu/chenhaoran/datasets/ground_truth_result'
        # self.save_path0_1 = '/hom e/libiao/数据/6.1混合数据/e0_1'
        # self.save_path1_1 = '/home/libiao/数据/6.1混合数据/e1_1'
        # self.save_path01 = '/home/libiao/数据/6.1混合数据/e01'
        # self.save_path10 = '/home/libiao/数据/6.1混合数据/e10'
        # self.save_path11 = '/home/libiao/数据/6.1混合数据/e11'
        # self.save_path_1_1 = '/home/libiao/数据/6.1混合数据/e_1_1'
        # self.save_path_10 = '/home/libiao/数据/6.1混合数据/e_10'
        # self.save_path_11 = '/home/libiao/数据/6.1混合数据/e_11'

        self.train_list = os.listdir(self.train_file)
        # self.double_edge_list = os.listdir(self.double_edge_file)
        self.double_edge_list =[]
        for item in self.train_list:
            temp = item.replace('Default', 'Gt')
            temp = temp.replace('png','bmp')
            temp = temp.replace('jpg', 'bmp')
            self.double_edge_list.append(temp)


        self.ground_list = []
        self.trainimage_list = []
        self.batch_size = batch_size_train
        self.gt_list = []
        self.train_image_list = []
        self.dou_edge_list = []
        self.final_dou_edge_list = []

        # random.shuffle(self.train_list)
        # random.shuffle(self.double_edge_list)
        # for edge_image in self.edge_list:
        #     # 在这个地方可以进行真实数据和合成数据的分离
        #     edge_image_split = edge_image.split('.')
        #     for train_image in self.train_list:
        #         # index = self.train_list.index(train_image)
        #         train_image_spilt = train_image.split('.')
        #         if edge_image_split[-2] == train_image_spilt[-2]:
        #             # print(train_image_spilt)
        #             # print(edge_image_split)
        #             self.trainimage_list.append(train_image)
        #             # self.dou_edge_list.append(edge_image)
        #             break

        # print(len(self.train_list))
        # for i in range(len(self.train_list)):
        #     print(i)
        #     filename1 = os.path.join(self.train_file, self.trainimage_list[i])
        #     # filename2 = os.path.join(self.edge_file, self.edge_list[i])
        #     filename3 = os.path.join(self.double_edge_file,self.dou_edge_list[i])
        #     self.train_image_list.append(filename1)
        #     # self.gt_list.append(filename2)
        #     self.final_dou_edge_list.append(filename3)
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train_image_list, self.gt_list,
        #                                                                         test_size=0.1, train_size=0.9,
        #                                                                         random_state=1300)

        for i in range(len(self.train_list)):
            train_filename = os.path.join(self.train_file, self.train_list[i])
            gt_filename = os.path.join(self.double_edge_file,self.double_edge_list[i])

            t1 = self.train_list[i]
            t1 = t1.replace('Default','')
            t1 = t1.replace('png','')
            t1 = t1.replace('jpg','')
            t1 = t1.replace('bmp','')
            t2 = self.double_edge_list[i]
            t2 = t2.replace('Gt', '')
            t2 = t2.replace('png', '')
            t2 = t2.replace('jpg', '')
            t2 = t2.replace('bmp', '')
            if t1!=t2:
                print(t1)
                print(t2)
                print('数据和GT不匹配')
                traceback.print_exc()

            self.train_image_list.append(train_filename)
            self.gt_list.append(gt_filename)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train_image_list, self.gt_list, test_size=0.1, train_size=0.9,random_state=1300)


        self.steps_per_epoch = len(self.X_train) / batch_size_train
        self.val_steps = len(self.X_test) / (batch_size_train)

        self.image_width = 320
        self.image_height = 320

        self.target_regression = True

    def get_batch(self, batch, train=True):

        filenames = []
        images = []
        edgemaps = []
        double_edge = []
        edgemaps_4 = []
        edgemaps_8 = []
        edgemaps_16 = []
        chanel1 = []
        chanel2 = []
        chanel3 = []
        chanel4 = []
        chanel5 = []
        chanel6 = []
        chanel7 = []
        chanel8 = []
        chanelfuse = []

        for idx, b in enumerate(batch):
            if train:
                index = self.X_train.index(b)
                im = Image.open(self.X_train[index])
                dou_path = os.path.join(self.double_edge_file, self.Y_train[index].split('/')[-1])
                dou_em = Image.open(dou_path)

                # path = os.path.join(self.save_path_1_1,self.Y_train[index].split('/')[-1])
                # chanel_1_1 = Image.open(path)
                # # -1 0
                # path = os.path.join(self.save_path_10,self.Y_train[index].split('/')[-1])
                # chanel_10 = Image.open(path)
                # # -1 1
                # path = os.path.join(self.save_path_11,self.Y_train[index].split('/')[-1])
                # chanel_11 = Image.open(path)
                # # 0 -1
                # path = os.path.join(self.save_path0_1,self.Y_train[index].split('/')[-1])
                # chanel0_1 = Image.open(path)
                # # 0 1
                # path = os.path.join(self.save_path01,self.Y_train[index].split('/')[-1])
                # chanel01 = Image.open(path)
                # # 1 -1
                # path = os.path.join(self.save_path1_1,self.Y_train[index].split('/')[-1])
                # chanel1_1 = Image.open(path)
                # # 1 0
                # path = os.path.join(self.save_path10,self.Y_train[index].split('/')[-1])
                # chanel10 = Image.open(path)
                # # 1 1
                # path = os.path.join(self.save_path11,self.Y_train[index].split('/')[-1])
                # chanel11 = Image.open(path)

                # 在这里获取8张图，从左上角按照顺时针顺序,返回的是一个长度为8的列表
                relation_8_map = gen_8_map(dou_em)
                for i in range(8):
                    relation_8_map[i] = Image.fromarray(np.uint8(relation_8_map[i]))


                weight = im.size[0]
                height = im.size[1]
                if im.size[0] > 320 and im.size[1] > 320:
                    w_centry = weight // 2
                    h_centry = height // 2

                    if w_centry > 160:
                        range_w = random.randint(0, w_centry - 160)
                    else:
                        range_w = 0
                    if h_centry > 160:
                        range_h = random.randint(0, h_centry - 160)
                    else:
                        range_h = 0

                    # 决定图像是否加压缩
                    if random.randint(0, 20) == 1:

                        try:
                            mb = random.randint(30, 100)
                            path = compress_image(infile=self.X_train[index], mb=mb)
                            im = Image.open(path)
                        except:
                            traceback.print_exc()
                    # 决定图像是否加模糊
                    if random.randint(0, 20) == 1:
                        m = random.randint(0, 5)
                        if m == 0:
                            r = random.randint(1, 3)
                            im = im.filter(ImageFilter.GaussianBlur(radius=r))
                        else:
                            pass

                    if random.randint(0, 1) == 0:
                        im = im.crop((w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                                      h_centry + 160 + range_h))
                        dou_em = dou_em.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].crop(
                                (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                                 h_centry + 160 + range_h))


                    else:
                        im = im.crop((w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                                      h_centry + 160 - range_h))

                        dou_em = dou_em.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        for i in range(8):
                            relation_8_map[i] = relation_8_map[i].crop(
                                (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                                 h_centry + 160 - range_h))

                        # 决定是否flip 旋转
                        random_aug = random.randint(0, 20)
                        try:
                            if random_aug < 5:
                                if random_aug == 0:
                                    # 旋转90
                                    im = im.transpose(Image.ROTATE_90)
                                    dou_em = dou_em.transpose(Image.ROTATE_90)
                                    for i in range(8):
                                        relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_90)

                                elif random_aug == 1:
                                    # 旋转180
                                    im = im.transpose(Image.ROTATE_180)
                                    dou_em = dou_em.transpose(Image.ROTATE_180)
                                    for i in range(8):
                                        relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_180)

                                elif random_aug == 2:
                                    # 旋转270
                                    im = im.transpose(Image.ROTATE_270)
                                    dou_em = dou_em.transpose(Image.ROTATE_270)
                                    for i in range(8):
                                        relation_8_map[i] = relation_8_map[i].transpose(Image.ROTATE_270)
                                elif random_aug == 3:
                                    # 左右呼唤
                                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                                    dou_em = dou_em.transpose(Image.FLIP_LEFT_RIGHT)
                                    for i in range(8):
                                        relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_LEFT_RIGHT)
                                elif random_aug == 4:
                                    # 左右呼唤
                                    im = im.transpose(Image.FLIP_TOP_BOTTOM)
                                    dou_em = dou_em.transpose(Image.FLIP_TOP_BOTTOM)
                                    for i in range(8):
                                        relation_8_map[i] = relation_8_map[i].transpose(Image.FLIP_TOP_BOTTOM)
                        except:
                            traceback.print_exc()
                else:
                    # 决定图像是否加压缩
                    if random.randint(0, 20) == 1:
                        # o_size = get_size(self.X_train[index])
                        try:
                            mb = random.randint(30, 100)
                            path = compress_image(infile=self.X_train[index], mb=mb)
                            im = Image.open(path)
                        except:
                            traceback.print_exc()
                    # 决定图像是否加模糊
                    if random.randint(0, 20) == 1:
                        m = random.randint(0, 5)
                        if m == 0:
                            r = random.randint(1, 3)
                            im = im.filter(ImageFilter.GaussianBlur(radius=r))
                        else:
                            pass
                            # x = random.randint(0, 200)
                            # y = random.randint(0, 200)
                            # bounds_x = random.randint(20, 100)
                            # bounds_y = random.randint(20, 100)
                            # bounds = (x, y, x + bounds_x, y + bounds_y)
                            # r = random.randint(1, 3)
                            # im = im.filter(MyGaussianBlur(radius=r, bounds=bounds))

                    im = im.crop((0, 0, self.image_height, self.image_width))

                    dou_em = dou_em.crop((0, 0, self.image_height, self.image_width))
                    for i in range(8):
                        relation_8_map[i] = relation_8_map[i].crop((0, 0, self.image_height, self.image_width))


                    # chanel_1_1 = chanel_1_1.crop((0, 0, self.image_height, self.image_width))
                    # chanel_10 = chanel_10.crop((0, 0, self.image_height, self.image_width))
                    # chanel_11 = chanel_11.crop((0, 0, self.image_height, self.image_width))
                    # chanel0_1 = chanel0_1.crop((0, 0, self.image_height, self.image_width))
                    # chanel01 = chanel01.crop((0, 0, self.image_height, self.image_width))
                    # chanel1_1 = chanel1_1.crop((0, 0, self.image_height, self.image_width))
                    # chanel10 = chanel10.crop((0, 0, self.image_height, self.image_width))
                    # chanel11 = chanel11.crop((0, 0, self.image_height, self.image_width))

                # print(im)
                im = np.array(im, dtype=np.float32)
                im = im[..., ::-1]  # RGB 2 BGR
                # R = im[..., 0].mean()
                # G = im[..., 1].mean()
                # B = im[..., 2].mean()
                # im[..., 0] -= R
                # im[..., 1] -= G
                # im[..., 2] -= B

                # R=118.98194217348079 G=127.4061956623793 B=138.00865419127499
                im[..., 0] -= 138.008
                im[..., 1] -= 127.406
                im[..., 2] -= 118.982

                dou_chanel = [0 for i in range(8)]
                chanel = [[] for i in range(8)]
                for i in range(8):
                    dou_chanel[i] = np.array(relation_8_map[i], dtype=np.float32)
                    dou_chanel[i] = np.array(dou_chanel[i][:, :, 1:])
                    dou_chanel[i] = dou_chanel[i] / 255
                    chanel[i].append(dou_chanel[i])

                c_1 = dou_chanel[0][:, :, 1:]
                c_2 = dou_chanel[1][:, :, 1:]
                c_3 = dou_chanel[2][:, :, 1:]
                c_4 = dou_chanel[3][:, :, 1:]
                c_5 = dou_chanel[4][:, :, 1:]
                c_6 = dou_chanel[5][:, :, 1:]
                c_7 = dou_chanel[6][:, :, 1:]
                c_8 = dou_chanel[7][:, :, 1:]
                final_c = np.concatenate((c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8), axis=2)
                chanelfuse.append(final_c)

                dou_em = np.array(dou_em, dtype=np.float32)
                # dou_em = np.array(dou_em[:, :, :])
                dou_em = np.array(dou_em[:, :])
                double_edge.append(dou_em)
                images.append(im)
                # edgemaps.append(bin_em)
                # edgemaps_4.append(bin_em4)
                # edgemaps_8.append(bin_em8)
                # edgemaps_16.append(bin_em16)
                filenames.append(self.X_train[index])

            else:
                index = self.X_test.index(b)
                im = Image.open(self.X_test[index])
                # 决定图像是否加压缩
                if random.randint(0, 20) == 1:
                    # o_size = get_size(self.X_train[index])
                    try:
                        mb = random.randint(30, 100)
                        path = compress_image(infile=self.X_test[index], mb=mb)
                        im = Image.open(path)
                    except:
                        traceback.print_exc()
                # 决定图像是否加模糊
                if random.randint(0, 20) == 1:
                    m = random.randint(0, 5)
                    if m == 0:
                        r = random.randint(1, 3)
                        im = im.filter(ImageFilter.GaussianBlur(radius=r))
                    else:
                        pass
                        # x = random.randint(0, 200)
                        # y = random.randint(0, 200)
                        # bounds_x = random.randint(20, 100)
                        # bounds_y = random.randint(20, 100)
                        # bounds = (x, y, x + bounds_x, y + bounds_y)
                        # r = random.randint(1, 3)
                        # im = im.filter(MyGaussianBlur(radius=r, bounds=bounds))
                em = Image.open(self.Y_test[index])
                # 这里不是dou_em了吗
                # 在这里获取8张图，从左上角按照顺时针顺序，返回的是一个长度为8的列表
                relation_8_map = gen_8_map(em)
                for i in range(8):
                    relation_8_map[i] = Image.fromarray(relation_8_map[i])
                # dou_path = os.path.join(self.double_edge_file, self.Y_test[index].split('/')[-1])
                # dou_em = Image.open(dou_path)
                #
                # path = os.path.join(self.save_path_1_1, self.Y_test[index].split('/')[-1])
                # chanel_1_1 = Image.open(path)
                # # -1 0
                # path = os.path.join(self.save_path_10, self.Y_test[index].split('/')[-1])
                # chanel_10 = Image.open(path)
                # # -1 1
                # path = os.path.join(self.save_path_11, self.Y_test[index].split('/')[-1])
                # chanel_11 = Image.open(path)
                # # 0 -1
                # path = os.path.join(self.save_path0_1, self.Y_test[index].split('/')[-1])
                # chanel0_1 = Image.open(path)
                # # 0 1
                # path = os.path.join(self.save_path01, self.Y_test[index].split('/')[-1])
                # chanel01 = Image.open(path)
                # # 1 -1
                # path = os.path.join(self.save_path1_1, self.Y_test[index].split('/')[-1])
                # chanel1_1 = Image.open(path)
                # # 1 0
                # path = os.path.join(self.save_path10, self.Y_test[index].split('/')[-1])
                # chanel10 = Image.open(path)
                # # 1 1
                # path = os.path.join(self.save_path11, self.Y_test[index].split('/')[-1])
                # chanel11 = Image.open(path)

                weight = im.size[0]
                height = im.size[1]
                if im.size[0] > 320 and im.size[1] > 320:
                    w_centry = weight // 2
                    h_centry = height // 2

                    if w_centry > 160:
                        range_w = random.randint(0, w_centry - 160)
                    else:
                        range_w = 0
                    if h_centry > 160:
                        range_h = random.randint(0, h_centry - 160)
                    else:
                        range_h = 0
                    if random.randint(0, 1) == 0:
                        im = im.crop((w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                                      h_centry + 160 + range_h))
                        em = em.crop((w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                                      h_centry + 160 + range_h))
                        dou_em = dou_em.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))

                        chanel_1_1 = chanel_1_1.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel_10 = chanel_10.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel_11 = chanel_11.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel0_1 = chanel0_1.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel01 = chanel01.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel1_1 = chanel1_1.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel10 = chanel10.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                        chanel11 = chanel11.crop(
                            (w_centry - 160 + range_w, h_centry - 160 + range_h, w_centry + 160 + range_w,
                             h_centry + 160 + range_h))
                    else:
                        im = im.crop((w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                                      h_centry + 160 - range_h))
                        em = em.crop((w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                                      h_centry + 160 - range_h))
                        dou_em = dou_em.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))

                        chanel_1_1 = chanel_1_1.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel_10 = chanel_10.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel_11 = chanel_11.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel0_1 = chanel0_1.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel01 = chanel01.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel1_1 = chanel1_1.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel10 = chanel10.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                        chanel11 = chanel11.crop(
                            (w_centry - 160 - range_w, h_centry - 160 - range_h, w_centry + 160 - range_w,
                             h_centry + 160 - range_h))
                else:
                    im = im.crop((0, 0, self.image_height, self.image_width))
                    em = em.crop((0, 0, self.image_height, self.image_width))
                    dou_em = dou_em.crop((0, 0, self.image_height, self.image_width))
                    chanel_1_1 = chanel_1_1.crop((0, 0, self.image_height, self.image_width))
                    chanel_10 = chanel_10.crop((0, 0, self.image_height, self.image_width))
                    chanel_11 = chanel_11.crop((0, 0, self.image_height, self.image_width))
                    chanel0_1 = chanel0_1.crop((0, 0, self.image_height, self.image_width))
                    chanel01 = chanel01.crop((0, 0, self.image_height, self.image_width))
                    chanel1_1 = chanel1_1.crop((0, 0, self.image_height, self.image_width))
                    chanel10 = chanel10.crop((0, 0, self.image_height, self.image_width))
                    chanel11 = chanel11.crop((0, 0, self.image_height, self.image_width))
                im = np.array(im, dtype=np.float32)
                im = im[..., ::-1]  # RGB 2 BGR
                # R=118.98194217348079 G=127.4061956623793 B=138.00865419127499
                im[..., 0] -= 138.008
                im[..., 1] -= 127.406
                im[..., 2] -= 118.982
                # im[..., 0] -= 103.939
                # im[..., 1] -= 116.779
                # im[..., 2] -= 123.68
                # R = im[..., 0].mean()
                # G = im[..., 1].mean()
                # B = im[..., 2].mean()
                # im[..., 0] -= 103.939
                # im[..., 1] -= 116.779
                # im[..., 2] -= 123.68
                # im[..., 0] -= R
                # im[..., 1] -= G
                # im[..., 2] -= B

                # Labels needs to be 1 or 0 (edge pixel or not)
                # or can use regression targets as done by the author
                # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213

                em_16 = em.resize((160, 160), Image.BICUBIC)
                em_16 = np.array(em_16.convert('L'), dtype=np.float32)
                em_16 = np.where(em_16 > 0, 255, 0)
                # new_image = Image.fromarray(np.uint8(em_16))

                em_8 = em.resize((80, 80), Image.BICUBIC)
                em_8 = np.array(em_8.convert('L'), dtype=np.float32)
                em_8 = np.where(em_8 > 0, 255, 0)
                # new_image = Image.fromarray(np.uint8(em_8))

                em_4 = em.resize((40, 40), Image.BICUBIC)
                em_4 = np.array(em_4.convert('L'), dtype=np.float32)
                em_4 = np.where(em_4 > 0, 255, 0)

                em = np.array(em.convert('L'), dtype=np.float32)

                if self.target_regression:
                    bin_em = em / 255.0
                else:
                    bin_em = np.zeros_like(em)
                    bin_em[np.where(em)] = 1
                bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
                bin_em = np.expand_dims(bin_em, 2)

                if self.target_regression:
                    bin_em4 = em_4 / 255.0
                else:
                    bin_em4 = np.zeros_like(em_4)
                    bin_em4[np.where(em_4)] = 1
                bin_em4 = bin_em4 if bin_em4.ndim == 2 else bin_em4[:, :, 0]
                bin_em4 = np.expand_dims(bin_em4, 2)

                if self.target_regression:
                    bin_em8 = em_8 / 255.0
                else:
                    bin_em8 = np.zeros_like(em_8)
                    bin_em8[np.where(em_8)] = 1
                bin_em8 = bin_em8 if bin_em8.ndim == 2 else bin_em8[:, :, 0]
                bin_em8 = np.expand_dims(bin_em8, 2)

                if self.target_regression:
                    bin_em16 = em_16 / 255.0
                else:
                    bin_em16 = np.zeros_like(em_16)
                    bin_em16[np.where(em_16)] = 1
                bin_em16 = bin_em16 if bin_em16.ndim == 2 else bin_em16[:, :, 0]
                bin_em16 = np.expand_dims(bin_em16, 2)

                dou_em = np.array(dou_em, dtype=np.float32)
                dou_em = np.array(dou_em[:, :, :])
                # dou_em = np.expand_dims(dou_em, 2)
                double_edge.append(dou_em)

                images.append(im)
                edgemaps.append(bin_em)
                edgemaps_16.append(bin_em16)
                edgemaps_8.append(bin_em8)
                edgemaps_4.append(bin_em4)
                filenames.append(self.X_test[index])

                # _1_1
                dou_chanel1 = np.array(chanel_1_1, dtype=np.float32)
                dou_chanel1 = np.array(dou_chanel1[:, :, 1:])
                dou_chanel1 = dou_chanel1 / 255
                chanel1.append(dou_chanel1)
                # chanel_1_1 = np.array(chanel_1_1.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel_1_1 = chanel_1_1 / 255.0
                # else:
                #     bin_chanel_1_1 = np.zeros_like(chanel_1_1)
                #     bin_chanel_1_1[np.where(chanel_1_1)] = 1
                # bin_chanel_1_1 = bin_chanel_1_1 if bin_chanel_1_1.ndim == 2 else bin_chanel_1_1[:, :, 0]
                # bin_chanel_1_1 = np.expand_dims(bin_chanel_1_1, 2)
                # chanel1.append(bin_chanel_1_1)

                # _1 0
                dou_chanel2 = np.array(chanel_10, dtype=np.float32)
                dou_chanel2 = np.array(dou_chanel2[:, :, 1:])
                dou_chanel2 = dou_chanel2 / 255
                chanel2.append(dou_chanel2)
                # chanel_10 = np.array(chanel_10.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel_10 = chanel_10 / 255.0
                # else:
                #     bin_chanel_10 = np.zeros_like(chanel_10)
                #     bin_chanel_10[np.where(chanel_10)] = 1
                # bin_chanel_10 = bin_chanel_10 if bin_chanel_10.ndim == 2 else bin_chanel_10[:, :, 0]
                # bin_chanel_10 = np.expand_dims(bin_chanel_10, 2)
                # chanel2.append(bin_chanel_10)

                # _1 1
                dou_chanel3 = np.array(chanel_11, dtype=np.float32)
                dou_chanel3 = np.array(dou_chanel3[:, :, 1:])
                dou_chanel3 = dou_chanel3 / 255
                chanel3.append(dou_chanel3)
                # chanel_11 = np.array(chanel_11.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel_11 = chanel_11 / 255.0
                # else:
                #     bin_chanel_11 = np.zeros_like(chanel_11)
                #     bin_chanel_11[np.where(chanel_11)] = 1
                # bin_chanel_11 = bin_chanel_11 if bin_chanel_11.ndim == 2 else bin_chanel_11[:, :, 0]
                # bin_chanel_11 = np.expand_dims(bin_chanel_11, 2)
                # chanel3.append(bin_chanel_11)

                # 0_1
                dou_chanel4 = np.array(chanel0_1, dtype=np.float32)
                dou_chanel4 = np.array(dou_chanel4[:, :, 1:])
                dou_chanel4 = dou_chanel4 / 255
                chanel4.append(dou_chanel4)
                # chanel0_1 = np.array(chanel0_1.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel0_1 = chanel0_1 / 255.0
                # else:
                #     bin_chanel0_1 = np.zeros_like(chanel0_1)
                #     bin_chanel0_1[np.where(chanel0_1)] = 1
                # bin_chanel0_1 = bin_chanel0_1 if bin_chanel0_1.ndim == 2 else bin_chanel0_1[:, :, 0]
                # bin_chanel0_1 = np.expand_dims(bin_chanel0_1, 2)
                # chanel4.append(bin_chanel0_1)

                # 0 1
                dou_chanel5 = np.array(chanel01, dtype=np.float32)
                dou_chanel5 = np.array(dou_chanel5[:, :, 1:])
                dou_chanel5 = dou_chanel5 / 255
                chanel5.append(dou_chanel5)
                # chanel01 = np.array(chanel01.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel01 = chanel01 / 255.0
                # else:
                #     bin_chanel01 = np.zeros_like(chanel01)
                #     bin_chanel01[np.where(chanel01)] = 1
                # bin_chanel01 = bin_chanel01 if bin_chanel01.ndim == 2 else bin_chanel01[:, :, 0]
                # bin_chanel01 = np.expand_dims(bin_chanel01, 2)
                # chanel5.append(bin_chanel01)

                # 1_1
                dou_chanel6 = np.array(chanel1_1, dtype=np.float32)
                dou_chanel6 = np.array(dou_chanel6[:, :, 1:])
                dou_chanel6 = dou_chanel6 / 255
                chanel6.append(dou_chanel6)
                # chanel1_1 = np.array(chanel1_1.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel1_1 = chanel1_1 / 255.0
                # else:
                #     bin_chanel1_1 = np.zeros_like(chanel1_1)
                #     bin_chanel1_1[np.where(chanel1_1)] = 1
                # bin_chanel1_1 = bin_chanel1_1 if bin_chanel1_1.ndim == 2 else bin_chanel1_1[:, :, 0]
                # bin_chanel1_1 = np.expand_dims(bin_chanel1_1, 2)
                # chanel6.append(bin_chanel1_1)

                # 1 0
                dou_chanel7 = np.array(chanel10, dtype=np.float32)
                dou_chanel7 = np.array(dou_chanel7[:, :, 1:])
                dou_chanel7 = dou_chanel7 / 255
                chanel7.append(dou_chanel7)
                # chanel10 = np.array(chanel10.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel10 = chanel10 / 255.0
                # else:
                #     bin_chanel10 = np.zeros_like(chanel10)
                #     bin_chanel10[np.where(chanel10)] = 1
                # bin_chanel10 = bin_chanel10 if bin_chanel10.ndim == 2 else bin_chanel10[:, :, 0]
                # bin_chanel10 = np.expand_dims(bin_chanel10, 2)
                # chanel7.append(bin_chanel10)

                # 1 1
                dou_chanel8 = np.array(chanel11, dtype=np.float32)
                dou_chanel8 = np.array(dou_chanel8[:, :, 1:])
                dou_chanel8 = dou_chanel8 / 255
                chanel8.append(dou_chanel8)

                c_1 = dou_chanel1[:, :, 1:]
                c_2 = dou_chanel2[:, :, 1:]
                c_3 = dou_chanel3[:, :, 1:]
                c_4 = dou_chanel4[:, :, 1:]
                c_5 = dou_chanel5[:, :, 1:]
                c_6 = dou_chanel6[:, :, 1:]
                c_7 = dou_chanel7[:, :, 1:]
                c_8 = dou_chanel8[:, :, 1:]
                final_c = np.concatenate((c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8), axis=2)
                # final_c = final_c.reshape(-1, 8)
                # print(final_c.shape)
                chanelfuse.append(final_c)
                # chanel11 = np.array(chanel11.convert('L'), dtype=np.float32)
                # if self.target_regression:
                #     bin_chanel11 = chanel11 / 255.0 1
                # else:
                #     bin_chanel11 = np.zeros_like(chanel11)
                #     bin_chanel11[np.where(chanel11)] = 1
                # bin_chanel11 = bin_chanel11 if bin_chanel11.ndim == 2 else bin_chanel11[:, :, 0]
                # bin_chanel11 = np.expand_dims(bin_chanel11, 2)
                # chanel8.append(bin_chanel11)

        images = np.asarray(images)
        # edgemaps = np.asarray(edgemaps)
        # edgemaps_4 = np.asanyarray(edgemaps_4)
        # edgemaps_8 = np.asanyarray(edgemaps_8)
        # edgemaps_16 = np.asanyarray(edgemaps_16)
        double_edge = np.asarray(double_edge)

        chanel1 = np.asarray(chanel[0])
        chanel2 = np.asarray(chanel[1])
        chanel3 = np.asarray(chanel[2])
        chanel4 = np.asarray(chanel[3])
        chanel5 = np.asarray(chanel[4])
        chanel6 = np.asarray(chanel[5])
        chanel7 = np.asarray(chanel[6])
        chanel8 = np.asarray(chanel[7])
        chanelfuse = np.asarray(chanelfuse)
        print('++++++++++++++++++++++++++++++++++++++')
        print(type(chanel1))
        print()
        print(chanel1.shape)

        return images, edgemaps, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanelfuse, edgemaps_4, edgemaps_8, edgemaps_16, filenames


def generate_minibatches(dataParser, train=True):
    print('123123')
    while True:
        if train:
            batch_ids = np.random.choice(dataParser.X_train, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids)
        else:
            batch_ids = np.random.choice(dataParser.X_test, dataParser.batch_size)
            ims, ems, double_edge, chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, chanel_fuse, edgemaps_4, edgemaps_8, edgemaps_16, _ = dataParser.get_batch(
                batch_ids, train=False)

        # datagen.flow()
        yield (ims, [chanel1, chanel2, chanel3, chanel4, chanel5, chanel6, chanel7, chanel8, ems, ems])



if __name__ == "__main__":
    # model
    dataParser = DataParser(6)

    try:
        t=generate_minibatches(dataParser=dataParser,train=True)
        print()
    except Exception as e:
        traceback.print_exc()