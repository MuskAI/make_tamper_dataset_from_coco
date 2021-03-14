"""
@author :haoran
time:0309
description:
改进F1,引入价值评价
"""
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,confusion_matrix
import numpy as np
from PIL import Image
import traceback

class Mymetrics:
    def __init__(self):
        pass
    def f1_score(self, pred, gt):
        label = gt.long()
        mask = (label != 0).float()
        num_positive = np.sum(mask).astype('float')
        num_negative = mask.numel() - num_positive
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)  # 0.995
        mask[mask == 0] = num_positive / (num_positive + num_negative)  # 0.005

        y = pred.reshape(-1)
        l = gt.reshape(-1)

        y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l.cpu().detach()).astype('int')

        return f1_score(y, l, zero_division=1)

        pass
    def value_f1_score(self, pred,label):
        mask = (label != 0)
        num_positive = np.sum(mask).astype('float')
        num_negative = mask - num_positive
        # print (num_positive, num_negative)
        w1 = num_negative / (num_positive + num_negative)
        w2 = num_positive / (num_positive + num_negative)
        mask = np.where(mask!=0,w1,w2)

        y = pred.reshape(-1)
        l = label.reshape(-1)
        mask = mask.reshape(-1)
        y = np.where(y > 0.5, 1, 0).astype('int')
        l = np.array(l).astype('int')
        result = confusion_matrix(y, l)
        print(result)

    def accuracy_score(self):
        pass
    def precision_score(self):
        pass
    def recall_score(self):
        pass

if __name__ == '__main__':
    pred = Image.open('./test/39t.bmp')
    pred = pred.split()[0]
    pred = np.array(pred)/255
    gt = Image.open('./test/39t_gt.bmp')
    gt = np.array(gt.split()[0])
    gt = np.where((gt == 255) | (gt == 100), 1, 0)

    Mymetrics().value_f1_score(pred,gt)