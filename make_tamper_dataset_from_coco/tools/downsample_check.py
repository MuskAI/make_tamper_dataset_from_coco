"""
created by haoran
time:8-24
检查各种下采样方法
"""
import numpy as np
from PIL import Image
import os,sys
import traceback

class DownSample():
    def __init__(self,image_path,save_path):
        self.image_path = image_path
        self.save_path = save_path

        pass
    def numpy_direct_resize(self):
        pass

    def Image_BiCubic_resize(self,target_size=(320,320),Binary_01 = False):


        img = Image.open(self.image_path)

        if self.required_condition(img.size,target_size) ==False:
            return False

        img_downsample = img.resize(target_size, Image.BICUBIC)
        if Binary_01 == False:
            img_downsample.save(self.save_path)
        else:
            img_downsample = np.where(np.array(img_downsample)>0,255,0)
            img_downsample = np.array(img_downsample,dtype='uint8')
            img_downsample = Image.fromarray(img_downsample).convert('RGB')
            img_downsample.save(self.save_path)
        print('双三次插值，保存在:',self.save_path)
        return True

    def required_condition(self,input_size,target_size):
        """
        判断尺寸大小合不合适
        :param input_size:
        :param target_size:
        :return:
        """
        if input_size[0]>target_size[0] and input_size[1] > target_size[1]:
            return True
        else:
            return False


if __name__ == '__main__':
    # image_path = 'H:\\TrainDataset\\GT\\camourflage_00328.png'
    # save_path = './11.bmp'
    img_root_path = 'H:\\TrainDataset\\GT'
    save_root_path = 'H:\\COD10K_resize\\GT'
    if os.path.exists(img_root_path):
        print(img_root_path,' 已经存在')
    else:
        print('数据集文件夹不存在，请检查')
        sys.exit()

    if os.path.exists(save_root_path):
        print(save_root_path, ' 已经存在')
    else:
        os.mkdir(save_root_path)

    size_error = []
    for index,img in enumerate(os.listdir(img_root_path)):
        img_path = os.path.join(img_root_path,img)

        save_path = os.path.join(save_root_path,img.replace('png','bmp').replace('jpg','bmp'))
        down = DownSample(img_path, save_path).Image_BiCubic_resize(Binary_01=True)
        if down == False:
            size_error.append(img)
            print(img,'大小不符合要求')
        print('the process:%d/%d'%(index,len(os.listdir(img_root_path))))
    print('%d张'%len(size_error),'不符合要求')
    print(size_error)








