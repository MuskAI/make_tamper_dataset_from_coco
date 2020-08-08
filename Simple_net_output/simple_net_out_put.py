from keras.layers import Conv2DTranspose,Conv2D, add,DepthwiseConv2D,Input, MaxPooling2D,BatchNormalization, Add,multiply
from keras.layers import Dropout, Concatenate,Activation, AveragePooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras.layers.core import  Lambda
from keras.initializers import glorot_uniform
import numpy as np
from keras.engine import Layer,InputSpec
from itertools import product
from subpixel_conv2d import SubpixelConv2D
from keras.utils import conv_utils
from keras.backend.common import normalize_data_format
# from DualAttention import CAM,PAM
from keras import initializers
def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)

    return x
def hed(input_shape=(320,320,3)):
    # Input
    img_input = Input(shape=input_shape, name='input')

    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x1 = Conv2D(64, (3, 3),  padding='same', name='block1_conv2')(x1)
    x1 = BatchNormalization(name='block1_conv2_bn')(x1)
    x1 = Activation("relu")(x1)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x1) # 240 240 64

    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x2)
    x2 = BatchNormalization(name='block2_conv2_bn')(x2)
    x2 = Activation("relu")(x2)
    b2= side_branch(x2, 2) # 480 480 1
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x2) # 120 120 128

    # Block 3
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x3)
    x3 = BatchNormalization(name='block3_conv3_bn')(x3)
    x3 = Activation("relu")(x3)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x3) # 60 60 256

    # Block 4
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x4)
    x4 = BatchNormalization(name='block4_conv3_bn')(x4)
    x4 = Activation("relu")(x4)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x4) # 30 30 512

    # Block 5
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x5)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x5) # 30 30 512
    x5 = BatchNormalization(name='block5_conv3_bn')(x5)
    x5 = Activation("relu")(x5)
    x5 =aspp(x5,input_shape,16)
    x5 =Conv2D(512, (1, 1), activation='relu', padding='same', name='aspp_conv1')(x5)



    # 20--40   512--128
    X_up=SubpixelConv2D(upsampling_factor=2, name='sub_pixel_1')(x5)
    X_up=Concatenate(axis=3)([x4,X_up])
    b4 = side_branch(X_up, 8)  # 320 320 1

    # 40--80   484--121
    X_up = Conv2D(512, (1, 1), activation='relu', padding='same', name='X_up_1')(X_up)
    X_up = SubpixelConv2D(upsampling_factor=2, name='sub_pixel_2')(X_up)
    X_up = Concatenate(axis=3)([x3, X_up])
    b3 = side_branch(X_up, 4)  # 480 480 1

    # 80--160
    X_up = Conv2D(128, (1, 1), activation='relu', padding='same', name='X_up_2')(X_up)
    X_up = SubpixelConv2D(upsampling_factor=2, name='sub_pixel_3')(X_up)
    X_up = Concatenate(axis=3)([x2, X_up])
    b2 = side_branch(X_up, 2)  # 480 480 1

    # 160--320 256---64
    X_up = Conv2D(128, (1, 1), activation='relu', padding='same', name='X_up_3')(X_up)
    X_up = SubpixelConv2D(upsampling_factor=2, name='sub_pixel_4')(X_up)
    final = Concatenate(axis=3)([x1, X_up])

    # 八通道先行训练
    X1 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel_1_1')(final)
    X2 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel_10')(final)
    X3 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel_11')(final)
    X4 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel0_1')(final)
    X5 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel01')(final)
    X6 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel1_1')(final)
    X7 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel10')(final)
    X8 = Conv2D(2, (3, 3), activation=None, padding='same', name='new1_e_chanel11')(final)


    # fuse
    fuse = Concatenate(axis=-1)([b2, b3, b4,X1,X2,X3,X4,X5,X6,X7,X8])
    fuse = Conv2D(1, (1, 1), padding='same', use_bias=False, activation=None)(fuse)  # 480 480 1

    # outputs

    o2    = Activation('sigmoid', name='o2')(b2)
    o3    = Activation('sigmoid', name='o3')(b3)
    o4    = Activation('sigmoid', name='o4')(b4)
    X1 = Activation('sigmoid', name='new_e_chanel_1_1')(X1)
    X2 = Activation('sigmoid', name='new_e_chanel_10')(X2)
    X3 = Activation('sigmoid', name='new_e_chanel_11')(X3)
    X4 = Activation('sigmoid', name='new_e_chanel0_1')(X4)
    X5 = Activation('sigmoid', name='new_e_chanel01')(X5)
    X6 = Activation('sigmoid', name='new_e_chanel1_1')(X6)
    X7 = Activation('sigmoid', name='new_e_chanel10')(X7)
    X8 = Activation('sigmoid', name='new_e_chanel11')(X8)
    fuse = Activation('sigmoid', name='fuse')(fuse)

    # model
    model = Model(inputs=[img_input], outputs=[ o2, o3, o4,X1,X2,X3,X4,X5,X6,X7,X8,fuse])

    model.summary()

    return model
def aspp(x, input_shape, out_stride):
    # 膨胀率3 6 9
    b0 = Conv2D(128, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    out_shape1 = int(input_shape[1] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape1))(x)
    b4 = Conv2D(128, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape1))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x
class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
            input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
            input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1]*self.upsampling[0]),
                                                   int(inputs.shape[2]*self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
if __name__ == '__main__':
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model=hed()