import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.layers import BatchNormalization, GlobalAveragePooling2D
from keras.layers import DepthwiseConv2D, concatenate, SeparableConv2D, add
from keras import Model
import numpy as np


class MnasNet():
    def __init__(self, batch_size, n_classes):
        self.model_name = 'se-net'
        self.data_size = [batch_size, 224, 224, 3]
        self.n_classes = n_classes
        lst = [1, 3, 3, 3, 2, 4, 1]
        self.lst = [1]
        for i in range(len(lst) + 1):
            if i == 0 or i == 1:
                continue
            self.lst.append(np.sum(lst[0:i]))  # [4, 7, 10, 12, 16, 17]
        self.decay_weight = 0.5e-3

    @property
    def img_size(self):
        return self.data_size[1:]

    def build(self):
        data = Input(batch_shape=self.data_size)
        net = Conv2D(32, [3, 3], strides=[2, 2], padding='same',
                     activation='relu', name='conv_0')(data)
        net = BatchNormalization(name='bn_0')(net)

        net = self._SepConv(net)

        for i in range(self.lst[0], self.lst[1]):
            net = self._MBConv3x3k_3(net, 24, i)
        net = MaxPooling2D([1, 1], strides=[2, 2], name='mp01')(net)

        for j in range(self.lst[1], self.lst[2]):
            net = self._MBConv5x5k_3(net, 40, j)
        net = MaxPooling2D([1, 1], strides=[2, 2], name='mp02')(net)

        for k in range(self.lst[2], self.lst[3]):
            net = self._MBConv5x5k_6(net, 80, k)
        net = MaxPooling2D([1, 1], strides=[2, 2], name='mp03')(net)

        for l in range(self.lst[3], self.lst[4]):
            net = self._MBConv3x3k_6(net, 96, l)

        for m in range(self.lst[4], self.lst[5]):
            net = self._MBConv5x5k_6(net, 192, m)
        net = MaxPooling2D([1, 1], strides=[2, 2], name='mp04')(net)

        for n in range(self.lst[5], self.lst[6]):
            net = self._MBConv3x3k_6(net, 320, n)

        net = Flatten()(net)
        net = Dense(self.n_classes, name='logits')(net)
        model = Model(inputs=data, outputs=net)
        model.summary()
        return net

    def ps(self, net):
        print(net.get_shape().as_list())

    def _SepConv(self, net):
        out_channels = net.get_shape().as_list()[-1] // 2
        print(out_channels)
        net = SeparableConv2D(out_channels, [3, 3], strides=[1, 1],
                              activation='relu', padding='same', name='SepConv_1_dwconv_1')(net)
        net = BatchNormalization(name='SepConv_1_bn_1')(net)
        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='same', name='SepConv_1_conv_2')(net)
        net = BatchNormalization(name='SepConv_1_bn_2')(net)
        return net

    def _MBConv3x3k_3(self, net, out_channels, indx):
        indx = str(indx)
        contract = net
        net = Conv2D(out_channels * 3, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv3x3k_3_{indx}_conv1')(net)
        net = BatchNormalization(name=f'MBConv3x3k_3_{indx}_bn1')(net)

        net = SeparableConv2D(out_channels * 3, [3, 3], strides=[1, 1], padding='same',
                              name=f'MBConv3x3k_3_{indx}_dwconv_2')(net)
        net = BatchNormalization(name=f'MBConv3x3k_3_{indx}_bn2')(net)

        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv3x3k_3_{indx}_conv3')(net)
        net = BatchNormalization(name=f'MBConv3x3k_3_{indx}_bn3')(net)

        if net.get_shape().as_list()[-1] == contract.get_shape().as_list()[-1]:
            # net = concatenate([net, contract], axis=-1)
            net = add([net, contract])
            return net
        else:
            return net

    def _MBConv3x3k_6(self, net, out_channels, indx):
        indx = str(indx)
        # out_channels = net.get_shape().as_list()[-1]
        contract = net

        net = Conv2D(out_channels * 6, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv3x3k_6_{indx}_conv1')(net)
        net = BatchNormalization(name=f'MBConv3x3k_6_{indx}_bn1')(net)

        net = SeparableConv2D(out_channels * 6, [3, 3], strides=[1, 1], padding='same',
                              name=f'MBConv3x3k_6_{indx}_dwconv_2')(net)
        net = BatchNormalization(name=f'MBConv3x3k_6_{indx}_bn2')(net)

        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv3x3k_6_{indx}_conv3')(net)
        net = BatchNormalization(name=f'MBConv3x3k_6_{indx}_bn3')(net)

        if net.get_shape().as_list()[-1] == contract.get_shape().as_list()[-1]:

            # net = concatenate([net, contract], axis=-1)
            net = add([net, contract])

            return net
        else:
            return net

    def _MBConv5x5k_3(self, net, out_channels, indx):
        indx = str(indx)
        contract = net
        net = Conv2D(out_channels * 3, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv5x5k_6_{indx}_conv1')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn1')(net)

        net = SeparableConv2D(out_channels * 3, [5, 5], strides=[1, 1], padding='same',
                              name=f'MBConv5x5k_6_{indx}_dwconv_2')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn2')(net)

        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv5x5k_6_{indx}_conv3')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn3')(net)
        if net.get_shape().as_list()[-1] == contract.get_shape().as_list()[-1]:
            # net = concatenate([net, contract], axis=-1)
            net = add([net, contract])
            return net
        else:
            return net

    def _MBConv5x5k_6(self, net, out_channels, indx):
        indx = str(indx)

        contract = net
        net = Conv2D(out_channels * 6, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv5x5k_6_{indx}_conv1')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn1')(net)

        net = SeparableConv2D(out_channels * 6, [5, 5], strides=[1, 1], padding='same',
                              name=f'MBConv5x5k_6_{indx}_dwconv_2')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn2')(net)

        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='valid',
                     activation='relu', name=f'MBConv5x5k_6_{indx}_conv3')(net)
        net = BatchNormalization(name=f'MBConv5x5k_6_{indx}_bn3')(net)
        if net.get_shape().as_list()[-1] == contract.get_shape().as_list()[-1]:
            # net = concatenate([net, contract], axis=-1)
            net = add([net, contract])
            return net
        else:
            return net


MnasNet(32, 100).build()
