from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import SeparableConv2D, DepthwiseConv2D,ReLU
from keras.layers import Dropout, BatchNormalization, Activation, concatenate
from keras import regularizers
from keras.models import Model
import keras.backend as K
import keras.activations as activation
from keras.utils.generic_utils import get_custom_objects
import numpy as np


# def relu6(x):
#     return K.minimum(K.maximum(0.0 ,x), 6.0)
#
# get_custom_objects().update({'relu6': Activation(relu6)})

class mobilenet_v1():
    def __init__(self, batch_size=32, n_classed=1000):
        self.input_shape = [batch_size, 224, 224, 3]
        self.n_classes = n_classed
        get_custom_objects().update({'relu6': Activation(self.relu6)})
    @property
    def img_size(self):
        return self.input_shape[1:]

    def relu6(self, x):
        return K.minimum(K.maximum(0.0, x), 6.0)


    def build(self):
        input_data = Input(batch_shape=self.input_shape)
        net = Conv2D(32, [3, 3], strides=[2, 2], padding='same', name='conv1_s2')(input_data)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv2_dw_s1')(net)
        # net = BatchNormalization()(net)
        net = Activation("relu6")(net)

        net = Conv2D(64, [1, 1], strides=[1, 1], padding='same', name='conv3_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(2, 2), padding='same', name='conv4_dw_s2')(net)


        net = Conv2D(128, [1, 1], strides=[1, 1], padding='same', name='conv5_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv6_dw_s1')(net)


        net = Conv2D(128, [1, 1], strides=[1, 1], padding='same', name='conv7_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(2, 2), padding='same', name='conv8_dw_s2')(net)

        net = Conv2D(128, [1, 1], strides=[1, 1], padding='same', name='conv9_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv10_dw_s1')(net)

        net = Conv2D(256, [1, 1], strides=[1, 1], padding='same', name='conv11_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(2, 2), padding='same', name='conv12_dw_s2')(net)


        net = Conv2D(256, [1, 1], strides=[1, 1], padding='same', name='conv13_s1')(net)


        net = Conv2D(512, [3, 3], strides=[1, 1], padding='same', name='conv14_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv15_dw_s1')(net)

        net = Conv2D(512, [3, 3], strides=[1, 1], padding='same', name='conv16_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv17_dw_s1')(net)


        net = Conv2D(512, [3, 3], strides=[1, 1], padding='same', name='conv18_s1')(net)
        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv19_dw_s1')(net)

        net = Conv2D(512, [3, 3], strides=[1, 1], padding='same', name='conv20_s1')(net)

        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv21_dw_s1')(net)
        net = Conv2D(512, [3, 3], strides=[1, 1], padding='same', name='conv22_s1')(net)


        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv23_dw_s1')(net)
        net = Conv2D(512, [1, 1], strides=[1, 1], padding='same', name='conv24_s1')(net)

        net = DepthwiseConv2D([3, 3], strides=(2, 2), padding='same', name='conv25_dw_s2')(net)
        net = Conv2D(1024, [1, 1], strides=[1, 1], padding='same', name='conv26_s1')(net)


        net = DepthwiseConv2D([3, 3], strides=(1, 1), padding='same', name='conv27_dw_s2')(net)

        net = Conv2D(1024, [1, 1], strides=[1, 1], padding='same', name='conv28_s1')(net)
        net = ReLU(max_value=6.0)(net)

        net = AvgPool2D([7,7],strides=(1,1),padding='valid',name='avg19_s1')(net)
        net = Flatten()(net)

        model = Model(inputs=input_data, outputs=net)
        model.summary()

if __name__ == '__main__':
    # get_custom_objects().update({'relu6': Activation(relu6)})
    mobilenet_v1().build()
