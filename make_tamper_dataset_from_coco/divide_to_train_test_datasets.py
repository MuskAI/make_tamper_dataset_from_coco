"""
created by hoaran
time : 2020-8-9
将数据集按照比例划分为训练集和测试集
"""
import random
import os
import sys
import datetime
import shutil
DATASET_SRC_PATH = 'C:\\Users\\musk\\Desktop\\test_cp\\tamper_poisson_result'
DATASET_TARGET_PATH = 'C:\\Users\\musk\\Desktop\\test_cp\\New3'

def divide(train_percent):
    if not os.path.exists(DATASET_SRC_PATH):
        print('DATASET_SRC_PATH错误，请确认输入的数据路径')
        sys.exit(1)
    if not os.path.exists(DATASET_TARGET_PATH):
        print('目标路径不存在，准备创建...')
        os.mkdir(DATASET_TARGET_PATH)
        os.mkdir(os.path.join(DATASET_TARGET_PATH, 'train_dataset_train_percent_%.2f@%d_%d'%(train_percent,
                                                                                                 datetime.datetime.now().month,
                                                                                                 datetime.datetime.now().day)))
        os.mkdir(os.path.join(DATASET_TARGET_PATH, 'test_dataset_train_percent_%.2f@%d_%d'%(train_percent,
                                                                                                 datetime.datetime.now().month,
                                                                                                 datetime.datetime.now().day)))
    else:
        print('目标路径存在，检查子文件夹')
        if os.path.exists(os.path.join(DATASET_TARGET_PATH, 'train_dataset_train_percent_%d@%d_%d'.format(train_percent,
                                                                                                          datetime.datetime.now().month,
                                                                                                          datetime.datetime.now().day))):
            print('数据集已经存在,请修改路径或者检查')
            sys.exit(1)

    data_list = os.listdir(DATASET_SRC_PATH)
    print('总共数据有：%d张'%len(data_list))
    print('训练:测试 = %d:%d'%(train_percent*10,10-train_percent))
    data_list = random.sample(data_list,len(data_list))
    train_set = random.sample(data_list,int(len(data_list) * train_percent))
    test_set = list(set(data_list).difference(set(train_set)))
    print(train_set)
    print(test_set)
    for index,train in enumerate(train_set):
        shutil.copy(os.path.join(DATASET_SRC_PATH,train),os.path.join(DATASET_TARGET_PATH,'train_dataset_train_percent_%.2f@%d_%d'%(train_percent,
                                                                                                 datetime.datetime.now().month,
                                                                                                 datetime.datetime.now().day)))
        print(index,'/',len(train_set))
    for index,test in enumerate(test_set):
        shutil.copy(os.path.join(DATASET_SRC_PATH,test),os.path.join(DATASET_TARGET_PATH,'test_dataset_train_percent_%.2f@%d_%d'%(train_percent,
                                                                                                 datetime.datetime.now().month,
                                                                                                 datetime.datetime.now().day)))

if __name__ == '__main__':
    divide(0.7)