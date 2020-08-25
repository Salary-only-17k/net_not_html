import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import Model
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import DepthwiseConv2D, concatenate, Flatten
import keras as k


class inception_resnet_v1():
    def __init__(self, batch_szie=32, n_classes=1000, weight_decay=0.00001, flags=1):
        self.data_shape = [batch_szie, 299, 299, 3]
        self.n_classes = n_classes
        self.flags = flags
        self.weight_decay = weight_decay
        self.model = 'inception_resnet_v1'

    def _conv_bn(self, net, n_out, k_szie, stride, pad):
        if isinstance(k_szie, int):
            net = Conv2D(n_out, [k_szie, k_szie],
                         strides=[stride, stride], padding=pad)(net)
            net = BatchNormalization()(net)
        else:

            net = Conv2D(n_out, k_szie,
                         strides=[stride, stride], padding=pad)(net)
            net = BatchNormalization()(net)
        return net

    def _conv_bn_relu(self, net, n_out, k_szie, stride, pad):
        if isinstance(k_szie, int):
            net = Conv2D(n_out, [k_szie, k_szie],
                         strides=[stride, stride], padding=pad)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
        else:
            net = Conv2D(n_out, k_szie,
                         strides=[stride, stride], padding=pad)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
        return net

    def _stem(self, net):
        net = self._conv_bn_relu(net=net, n_out=32, k_szie=3, stride=2, pad='valid')
        net = self._conv_bn_relu(net=net, n_out=32, k_szie=3, stride=1, pad='valid')
        net = self._conv_bn_relu(net=net, n_out=64, k_szie=3, stride=1, pad='same')
        net = MaxPooling2D([3, 3], strides=[2, 2], padding='valid')(net)
        net = self._conv_bn_relu(net=net, n_out=80, k_szie=1, stride=1, pad='same')
        net = self._conv_bn_relu(net=net, n_out=192, k_szie=3, stride=1, pad='valid')
        net = self._conv_bn_relu(net=net, n_out=256, k_szie=3, stride=2, pad='valid')
        return net

    def _reductionA(self, net, out_lst):
        net1 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid')(net)

        net2 = self._conv_bn_relu(net, n_out=out_lst[0], k_szie=3, stride=2, pad='valid')

        net3 = self._conv_bn_relu(net, n_out=out_lst[1], k_szie=1, stride=1, pad='same')
        net3 = self._conv_bn_relu(net3, n_out=out_lst[2], k_szie=3, stride=1, pad='same')
        net3 = self._conv_bn_relu(net3, n_out=out_lst[3], k_szie=3, stride=2, pad='valid')
        return concatenate([net1, net2, net3], axis=-1)

    def _inception_resnetA(self, net):

        # print('residual3.  ', residual3.get_shape().as_list())
        net0 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same')

        net1 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same')
        net1 = self._conv_bn_relu(net=net1, n_out=32, k_szie=3, stride=1, pad='same')

        net2 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same')
        net2 = self._conv_bn_relu(net=net2, n_out=32, k_szie=3, stride=1, pad='same')
        net2 = self._conv_bn_relu(net=net2, n_out=32, k_szie=3, stride=1, pad='same')

        net3 = concatenate([net1, net2, net0], axis=3)
        net3 = self._conv_bn(net=net3, n_out=256, k_szie=1, stride=1, pad='same')

        # print('net  ',net.get_shape().as_list())
        net4 = Add()([net3, net])
        return net4

    def _inception_resnetB(self, net):

        net0 = self._conv_bn_relu(net=net, n_out=128, k_szie=1, stride=1, pad='same')

        net1 = self._conv_bn_relu(net=net, n_out=128, k_szie=1, stride=1, pad='same')
        net1 = self._conv_bn_relu(net=net1, n_out=128, k_szie=[1, 7], stride=1, pad='same')
        net1 = self._conv_bn_relu(net=net1, n_out=128, k_szie=[7, 1], stride=1, pad='same')

        net3 = concatenate([net1, net0], axis=-1)

        net3 = self._conv_bn(net=net3, n_out=896, k_szie=1, stride=1, pad='same')

        net4 = Add()([net3, net])
        return Activation('relu')(net4)

    def _reductionB(self, net):

        net1 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid')(net)

        net2 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same')
        net2 = self._conv_bn_relu(net=net2, n_out=384, k_szie=3, stride=2, pad='valid')

        net3 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same')
        net3 = self._conv_bn_relu(net=net3, n_out=256, k_szie=3, stride=2, pad='valid')

        net4 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same')
        net4 = self._conv_bn_relu(net=net4, n_out=256, k_szie=3, stride=1, pad='same')
        net4 = self._conv_bn_relu(net=net4, n_out=256, k_szie=3, stride=2, pad='valid')

        net = concatenate([net4, net1, net2, net3],axis=-1)
        return net

    def _inception_resnetC(self, net):

        net1 = self._conv_bn_relu(net=net, n_out=192, k_szie=1, stride=1, pad='same')

        net2 = self._conv_bn_relu(net=net, n_out=192, k_szie=1, stride=1, pad='same')
        net2 = self._conv_bn_relu(net=net2, n_out=192, k_szie=[1, 3], stride=1, pad='same')
        net2 = self._conv_bn_relu(net=net2, n_out=192, k_szie=[3, 1], stride=1, pad='same')

        net3 = concatenate([net2, net1], axis=-1)
        net3 = self._conv_bn(net=net3, n_out=1792, k_szie=1, stride=1, pad='same')
        net4 = Add()([net3, net])
        return Activation('relu')(net4)

    def build_inception_resnet_a(self, net, n_itsrs):
        for _ in range(n_itsrs):
            self._inception_resnetA(net=net)
        return net

    def build_inception_resnet_b(self, net, n_itsrs):
        for _ in range(n_itsrs):
            self._inception_resnetB(net=net)
        return net

    def build_inception_resnet_c(self, net, n_itsrs):
        for _ in range(n_itsrs):
            self._inception_resnetC(net=net)
        return net

    def build(self):
        data = Input(batch_shape=self.data_shape)
        net = self._stem(data)

        net = self.build_inception_resnet_a(net=net, n_itsrs=5)
        # print('build_inception_resnet_a ',net.get_shape().as_list())
        net = self._reductionA(net=net, out_lst=[384, 192, 192, 256])
        # print('_reductionA  ',net.get_shape().as_list())

        net = self.build_inception_resnet_b(net, 10)
        # print('build_inception_resnet_b  ',net.get_shape().as_list())
        net = self._reductionB(net)
        # print('_reductionB  ', net.get_shape().as_list())

        net = self.build_inception_resnet_c(net, 5)
        # print('_reductionC  ', net.get_shape().as_list())

        net = AveragePooling2D([8, 8], strides=[1, 1], padding='valid')(net)
        # print('AveragePooling2D  ', net.get_shape().as_list())
        net = Dropout(0.8)(net)
        net = Flatten()(net)
        net = Dense(self.n_classes, activation='softmax')(net)
        # print('Dense  ', net.get_shape().as_list())
        model = Model(inputs=data, outputs=net,name=self.model)
        model.summary()
        return net


inception_resnet_v1().build()
