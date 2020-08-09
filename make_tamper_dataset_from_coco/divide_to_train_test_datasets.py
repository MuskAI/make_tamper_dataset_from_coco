"""
created by hoaran
time : 2020-8-9
将数据集按照比例划分为训练集和测试集
"""
import random
import os
import sys
import datetime
DATASET_SRC_PATH = ''
DATASET_TARGET_PATH = ''

def divide(train_percent):
    if not os.path.exists(DATASET_SRC_PATH):
        print('DATASET_SRC_PATH错误，请确认输入的数据路径')
        sys.exit(1)
    if not os.path.exists(DATASET_TARGET_PATH):
        print('目标路径不存在，准备创建...')
        os.mkdir(DATASET_TARGET_PATH)
        os.mkdir(os.path.join(DATASET_TARGET_PATH, 'train_dataset_train_percent_%d@%d_%d'.format(train_percent,
                                                                                                 datetime.datetime.now().month,
                                                                                                 datetime.datetime.now().day)))
        os.mkdir(os.path.join(DATASET_TARGET_PATH, 'test_dataset_train_percent_%d@%d_%d'.format(train_percent,
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
    print('总共数据有：%d张',len(data_list))
    print('训练:测试 = %d:%d'.format(train_percent,10-train_percent))
    random.sample(data_list,len(data_list))
    train_set = random.sample(data_list,len(data_list) * train_percent)
    test_set = list(set(data_list).difference(set(train_set)))