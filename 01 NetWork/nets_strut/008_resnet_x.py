import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate, Lambda
from keras import regularizers
from keras.models import Model
import keras.backend as K
from keras.layers.merge import add
'''
chenglong
201912
'''

def _bn_relu(net):
    net = BatchNormalization()(net)
    return Activation("relu")(net)


def _conv_bn_relu(n_out, kernel_size, strides=1, pad='same'):
    def inner(net):
        net = Conv2D(n_out, kernel_size,
                     strides=strides, padding=pad,
                     kernel_initializer="he_normal",
                     kernel_regularizer=regularizers.l2(1.e-4))(net)
        return _bn_relu(net)

    return inner


def _bn_relu_conv(n_out, kernel_size, strides=1, pad='same'):
    def inner(net):
        net = _bn_relu(net)
        return Conv2D(n_out, kernel_size,
                      strides=strides, padding=pad,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(1.e-4))(net)

    return inner


def _shortcut(inputs, residual):
    inputs_shape = inputs.get_shape().as_list()
    residual_shape = residual.get_shape().as_list()
    stride_width = int(round(inputs_shape[1] / residual_shape[1]))
    stride_height = int(round(inputs_shape[2] / residual_shape[2]))
    equal_channels = inputs_shape[3] == residual_shape[3]

    shortcut = inputs
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=regularizers.l2(0.0001))(inputs)

    return add([shortcut, residual])


def _residual_block(block_func, n_out, repetitions, is_first_layer=False):
    def inner(net):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            net = block_func(n_out=n_out, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(net)
        return net

    return inner


def basic_block(n_out, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def inner(input):
        if is_first_block_of_first_layer:
            conv1 = Conv2D(n_out, (3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(n_out=n_out, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(n_out=n_out, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return inner


def bottleneck(n_out, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def inner(input):

        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters=n_out, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=regularizers.l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(n_out=n_out, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(n_out=n_out, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(n_out=n_out * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return inner


class ResnetBuilder(object):
    def __init__(self, batch_size, num_outputs, block_fn, repetitions):
        self.data_size = [batch_size,224,224,3]
        self.num_outputs = num_outputs
        self.block_fn = block_fn
        self.repetitions = repetitions

    @property
    def input_size(self):
        return self.data_size[1:]
    def build(self):
        data = Input(batch_shape=self.data_size)
        net = _conv_bn_relu(n_out=64, kernel_size=(7, 7), strides=(2, 2))(data)
        net = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same")(net)

        n_out = 64
        for i, r in enumerate(self.repetitions):
            net = _residual_block(self.block_fn, n_out=n_out, repetitions=r, is_first_layer=(i == 0))(net)
            n_out *= 2  # 下一个repeation 是前一个的二倍

        net = _bn_relu(net)
        block_shape = net.get_shape().as_list()
        net = AvgPool2D(pool_size=(block_shape[1:3]),
                          strides=(1, 1))(net)
        net = Flatten()(net)
        net = Dense(units=self.num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(net)

        model = Model(inputs=data, outputs=net)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder(input_shape, num_outputs, basic_block, [2, 2, 2, 2]).build()

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder(input_shape, num_outputs, basic_block, [3, 4, 6, 3]).build()

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder(input_shape, num_outputs, bottleneck, [3, 4, 6, 3]).build()

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder(input_shape, num_outputs, bottleneck, [3, 4, 23, 3]).build()

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder(input_shape, num_outputs, bottleneck, [3, 8, 36, 3]).build()


ResnetBuilder.build_resnet_18(32, 1000).summary()
