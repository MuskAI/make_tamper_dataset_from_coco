import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# loss function
def sigmoid_cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

def weighted_nll_loss(prediction, label):
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = torch.nn.functional.nll_loss(prediction, label, reduce=False)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    return torch.sum(cost)

def weighted_cross_entropy_loss(prediction, label, output_mask=False):
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = criterion(prediction, label)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    if output_mask:
        return torch.sum(cost), (label != 0)
    else:
        return torch.sum(cost)

def l2_regression_loss(prediction, label, mask):
    label = torch.squeeze(label.float())
    prediction = torch.squeeze(prediction.float())
    mask = (mask != 0).float()
    num_positive = torch.sum(mask).float()
    cost = torch.nn.functional.mse_loss(prediction, label, reduce=False)
    cost = torch.mul(cost, mask)
    cost = cost / (num_positive + 0.00000001)
    return torch.sum(cost)

def l1_loss(prediction,label):
    return torch.nn.L1Loss()(prediction,label)


def smooth_l1_loss(prediction, label):
    return torch.nn.SmoothL1Loss()(prediction,label)




def my_accuracy_score(prediction,label):
    prediction = np.squeeze(prediction)
    label = np.squeeze(label)
    batch_size = prediction.shape[0]

    acc = 0
    for i in range(batch_size):
        prediction[i,:,:] = np.where(prediction[i,:,:] >0.6,1,0)
        acc += accuracy_score(prediction[i,:,:],label[i,:,:])
    return acc

def my_precision_score(prediction,label):
    return precision_score(prediction,label)

def my_f1_score(prediction,label):
    return f1_score(prediction,label)