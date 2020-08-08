import tensorflow as tf
from keras import backend as K
from keras.losses import mean_absolute_error
import numpy as np

import tensorflow.contrib.slim as slim
import math
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


def gradient_1order(x):
    r = tf.pad(x,paddings=[[0,0],[0,0],[0,1],[0,0]],mode='CONSTANT')[:, :, 1:,:]
    l = tf.pad(x, paddings=[[0,0],[0, 0],[1, 0],[0,0]], mode='CONSTANT')[:, :, :320, :]
    t = tf.pad(x , paddings=[[0,0],[1, 0], [0, 0],[0,0]], mode='CONSTANT')[:,:320, :, :]
    b = tf.pad(x, paddings=[[0,0],[0, 1], [0, 0],[0,0]], mode='CONSTANT')[:,1:, :, :]
    grad = tf.abs(r - l) + tf.abs(t - b)
    return grad


def gradLoss(y_true, y_pred):
    igrad = gradient_1order(y_true)
    tgrad = gradient_1order(y_pred)
    return 1000*tf.reduce_mean(tf.abs(igrad-tgrad))
    
def cross_entropy_loss_RCF(y_true, y_pred):
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    # grad=gradLoss(y_true,y_pred)
    # costSSIM = SSIM(y_true, y_pred)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)
    lamda = 1.2
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    alpha = lamda * count_pos / (count_neg + count_pos)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / alpha

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = 1000*tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
def spatialloss(y_true,y_pred,len_reg=0):
    vec1=tf.reshape(y_true,[-1,8])
    vec2=tf.reshape(y_pred,[-1,8])
    clip_value=0.999999
    norm_vec1=tf.nn.l2_normalize(vec1,1)
    norm_vec2=tf.nn.l2_normalize(vec2,1)
    dot = tf.reduce_sum(norm_vec1*norm_vec2,1)
    dot = tf.clip_by_value(dot,-clip_value,clip_value)
    angle=tf.acos(dot)*(180/math.pi)
    return tf.reduce_mean(angle)

def focal_loss_fixed(y_true, y_pred,gamma=2., alpha=.25):
    y_pred1=tf.slice(y_pred,[0,0,0,0],[-1,-1,-1,1])
    y_pred2=tf.slice(y_pred,[0,0,0,1],[-1,-1,-1,16])
    y_true1=tf.slice(y_true,[0,0,0,0],[-1,-1,-1,1])
    y_true2=tf.slice(y_true,[0,0,0,1],[-1,-1,-1,16])

    _epsilon = _to_tensor(K.epsilon(), y_pred2.dtype.base_dtype)
    y_pred2 = tf.clip_by_value(y_pred2, _epsilon, 1 - _epsilon)
    y_pred2 = tf.log(y_pred2 / (1 - y_pred2))
    y_true2 = tf.cast(y_true2, tf.float32)
    count_neg = tf.reduce_sum(1. - y_true2)
    count_pos = tf.reduce_sum(y_true2)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred2, targets=y_true2, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))

    count_neg1 = tf.reduce_sum(1. - y_true1)
    count_pos1 = tf.reduce_sum(y_true1)
    beta1 = count_neg1 / (count_neg1 + count_pos1)
    pos_weight1 = beta1 / (1 - beta1)

    y_pred21_1, y_pred21_0 = slice(y_pred2, 0)
    y_pred22_1, y_pred22_0 = slice(y_pred2, 2)
    y_pred23_1, y_pred23_0 = slice(y_pred2, 4)
    y_pred24_1, y_pred24_0 = slice(y_pred2, 6)
    y_pred25_1, y_pred25_0 = slice(y_pred2, 8)
    y_pred26_1, y_pred26_0 = slice(y_pred2, 10)
    y_pred27_1, y_pred27_0 = slice(y_pred2, 12)
    y_pred28_1, y_pred28_0 = slice(y_pred2, 14)

    y_same = y_pred21_1 + y_pred22_1 + y_pred23_1 + y_pred24_1 + y_pred25_1 + y_pred26_1 + y_pred27_1 + y_pred28_1
    y_diff = y_pred21_0 + y_pred22_0 + y_pred23_0 + y_pred24_0 + y_pred25_0 + y_pred26_0 + y_pred27_0 + y_pred28_0

    y_same = tf.cast(tf.greater(y_same, 1.9), tf.float32)
    y_diff = tf.cast(tf.greater(y_diff, 0.9), tf.float32)
    y_all = tf.cast(tf.greater(y_diff + y_same, 0.9), tf.float32)+1
    pt_1 = tf.where(tf.equal(y_true1, 1), y_pred1, tf.ones_like(y_pred1))
    pt_0 = tf.where(tf.equal(y_true1, 0), y_pred1, tf.zeros_like(y_pred1))
    cost1= -K.sum(alpha * K.pow(1. - pt_1, y_all) * K.log(K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, y_all) * K.log(1. - pt_0 + K.epsilon()))
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #     # sess.run(pred1)
    #     print(cost.eval())
    return cost
    # pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    # pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum(
    #     (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
# 切片统计
def slice(y_pred,index1,index2=2):
    y_pred = tf.slice(y_pred, [0, 0, 0, index1], [-1, -1, -1, index2])
    # 先转化为 0 1
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
    pred1=tf.slice(pred, [0, 0, 0, 0], [-1, -1, -1, 1])
    pred2=tf.slice(pred, [0, 0, 0, 1], [-1, -1, -1, 1])
    # 先判断 1 1的个数 边缘同类点
    pred3=tf.cast(tf.equal(pred1,pred2), tf.float32)
    pred3=tf.cast(tf.equal(pred1,pred3),tf.float32)
    # 边缘不同类点 1 0
    pred4=tf.cast(tf.not_equal(pred1,pred2),tf.float32)
    pred4=tf.cast(tf.equal(pred1,pred4),tf.float32)

    return pred3,pred4
def pixel_error(y_true, y_pred):

    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')
def real_cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    # 这里的y_pred的第一个通道都是一层模板（表示该点是否计算loss）
    # 首先将y_pred的第一个通道拿出来，然后将其作为模板乘到计算之后的loss上去

    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    # Equation [2]
    beta = count_neg / (count_neg + count_pos)
    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))
    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN


def acc(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC
def precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP/(TP + FP + K.epsilon())
    return PC


def sensitivity(y_true, y_pred):
    """ recall """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP/(TP + FN + K.epsilon())
    return SE





def specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP


def f1_socre(y_true, y_pred):
    SE = sensitivity(y_true, y_pred)
    PC = precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1



def huber_loss(y_true,y_pred):
    return tf.losses.huber_loss(labels=y_true,predictions=y_pred)

def Weight_Mse_loss(y_true,y_pred):
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    y_weight =y_true
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    y_weight=pos_weight*y_weight
    cost=K.mean(y_weight*K.square(y_true-y_pred))
    return cost
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def wce_huber_loss(y_true,y_pred,e1=0.45,e2=0.05):
    loss1=cross_entropy_balanced(y_true,y_pred)
    loss2=huber_loss(y_true,y_pred)
    loss3=dice_coef_loss(y_true,y_pred)
    return (1-e1-e2)*loss1+e1*loss2+e2*loss3

def wce_huber_loss_eight(y_true,y_pred,e1=0.5):
    loss1=cross_entropy_balanced(y_true,y_pred)
    loss2=huber_loss(y_true,y_pred)
    return e1*loss1+(1-e1)*loss2
def wce_huber_loss_new(y_true,y_pred,e1=0.7):
    loss1=cross_entropy_balanced(y_true,y_pred)
    loss2=huber_loss(y_true,y_pred)
    return e1*loss1+(1-e1)*loss2
def Wce_L1_cross_entropy_balanced(y_true, y_pred):
    loss1 = cross_entropy_balanced(y_true, y_pred)

    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    # Equation [2]
    beta = count_neg / (count_neg + count_pos)
    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    return
def cross_entropy_balanced(y_true, y_pred):
    """
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)
    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    # Equation [2]
    beta = count_neg / (count_neg + count_pos)
    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta))
    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
# 没用
def cross_entropy_balanced1(y_true, y_pred):
    """
        Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
        Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
        """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.log(y_pred / (1 - y_pred))
    y_true = tf.cast(y_true, tf.float32)

    y_pred1 = tf.slice(y_pred, [0, 0, 0, 0], [-1, -1, -1, 1])
    cost=tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=1)
    cost=cost*y_pred1
    cost = tf.reduce_mean(cost)

    return cost


def double_pixel_loss(y_true,y_pred):

    return



def _tf_fspecial_gauss(size, sigma=1.5):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def SSIM(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """

    # img1 = tf.expand_dims(img1, 0)
    # img1 = tf.expand_dims(img1, -1)
    # img2 = tf.expand_dims(img2, 0)
    # img2 = tf.expand_dims(img2, -1)
    window = _tf_fspecial_gauss(window_size)

    # _epsilon = _to_tensor(K.epsilon(), img2.dtype.base_dtype)
    # y_pred = tf.clip_by_value(img2, _epsilon, 1 - _epsilon)
    # y_pred = tf.log(y_pred / (1 - y_pred))
    # y_true = tf.cast(img1, tf.float32)


    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')


    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    ssim_map = 1-((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    return tf.reduce_mean(ssim_map)

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


if __name__ == '__main__':
    import tensorflow as tf
    from numpy.random import RandomState

    """
    自定义损失函数 
    """
    batch_size = 8

    x = tf.placeholder(tf.float32, shape=(None, 2), name='x_input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')  # 真值

    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w1)  # 预测值

    loss_less = 10
    loss_more = 1

    loss = cross_entropy_balanced(x,y)

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    rdm = RandomState(1)

    dataset_size = 128
    X = rdm.rand(dataset_size, 2)

    Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        print("a: ", sess.run(tf.shape(x)), x)
        STEPS = 5000
        for i in range(STEPS):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            # 通过选取的样本训练神经网络或训练参数
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        print(sess.run(w1))

