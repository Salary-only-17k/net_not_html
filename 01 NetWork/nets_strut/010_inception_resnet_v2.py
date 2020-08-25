import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import Model
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import DepthwiseConv2D, concatenate, Flatten
import keras as k


class inception_resnet_v2():
    def __init__(self, batch_szie=32, n_classes=1000, weight_decay=0.00001, flags=1):
        self.data_shape = [batch_szie, 299, 299, 3]
        self.n_classes = n_classes
        self.flags = flags
        self.weight_decay = weight_decay
        self.model = 'inception_resnet_v1'

    def _conv_bn(self, layer_name, net, n_out, k_szie, stride, pad):
        if isinstance(k_szie, int):
            net = Conv2D(n_out, [k_szie, k_szie],
                         strides=[stride, stride], padding=pad, name=layer_name)(net)
            net = BatchNormalization()(net)
        else:

            net = Conv2D(n_out, k_szie,
                         strides=[stride, stride], padding=pad, name=layer_name)(net)
            net = BatchNormalization()(net)
        return net

    def _conv_bn_relu(self, layer_name, net, n_out, k_szie, stride, pad):
        if isinstance(k_szie, int):
            net = Conv2D(n_out, [k_szie, k_szie],
                         strides=[stride, stride], padding=pad, name=layer_name)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
        else:
            net = Conv2D(n_out, k_szie,
                         strides=[stride, stride], padding=pad, name=layer_name)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
        return net

    def _stem(self, net):
        net0 = self._conv_bn_relu('stem_conv1', net=net, n_out=32, k_szie=3, stride=2, pad='valid')
        net0 = self._conv_bn_relu('stem_conv2', net=net0, n_out=32, k_szie=3, stride=1, pad='valid')
        net0 = self._conv_bn_relu('stem_conv3', net=net0, n_out=64, k_szie=3, stride=1, pad='same')
        # print(net0.get_shape().as_list())

        net1 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid', name='stem_b1')(net0)
        # print(net1.get_shape().as_list())
        net2 = self._conv_bn_relu('stem_b2', net=net0, n_out=96, k_szie=3, stride=2, pad='valid')
        # print(net2.get_shape().as_list())
        netx = concatenate([net2, net1], axis=-1)
        print(netx.get_shape().as_list())

        net3 = self._conv_bn_relu('stem_b3_1', net=netx, n_out=64, k_szie=1, stride=1, pad='same')
        net3 = self._conv_bn_relu('stem_b3_2', net=net3, n_out=96, k_szie=3, stride=1, pad='valid')

        net4 = self._conv_bn_relu('stem_b4_1', net=netx, n_out=64, k_szie=1, stride=1, pad='same')
        net4 = self._conv_bn_relu('stem_b4_2', net=net4, n_out=64, k_szie=[7, 1], stride=1, pad='same')
        net4 = self._conv_bn_relu('stem_b4_3', net=net4, n_out=64, k_szie=[1, 7], stride=1, pad='same')
        net4 = self._conv_bn_relu('stem_b4_4', net=net4, n_out=96, k_szie=3, stride=1, pad='valid')

        netx = concatenate([net4, net3], axis=-1)
        net5 = self._conv_bn_relu('stem_b5_1', net=netx, n_out=192, k_szie=3, stride=2, pad='valid')
        net6 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid', name='stem_b6_1')(netx)
        netx = concatenate([net6, net5], axis=-1)
        print(netx.get_shape().as_list())
        return netx

    def _inception_resnetA(self,layer_id, net):

        # print('residual3.  ', residual3.get_shape().as_list())
        net0 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b1_1')

        net1 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b2_1')
        net1 = self._conv_bn_relu(net=net1, n_out=32, k_szie=3, stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b2_2')

        net2 = self._conv_bn_relu(net=net, n_out=32, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b3_1')
        net2 = self._conv_bn_relu(net=net2, n_out=48, k_szie=3, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b3_2')
        net2 = self._conv_bn_relu(net=net2, n_out=64, k_szie=3, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b3_3')

        net3 = concatenate([net1, net2, net0], axis=3)
        net3 = self._conv_bn(net=net3, n_out=384, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetA_{layer_id}_b4_1')

        # print('net  ',net.get_shape().as_list())
        net4 = Add()([net3, net])
        return net4

    def _reductionA(self, net, out_lst):
        net1 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid', name='reductionA_b1')(net)

        net2 = self._conv_bn_relu('reductionA_b2', net=net, n_out=out_lst[0], k_szie=3, stride=2, pad='valid')

        net3 = self._conv_bn_relu('reductionA_b3_1', net=net, n_out=out_lst[1], k_szie=1, stride=1, pad='same')
        net3 = self._conv_bn_relu('reductionA_b3_2', net=net3, n_out=out_lst[2], k_szie=3, stride=1, pad='same')
        net3 = self._conv_bn_relu('reductionA_b3_3', net=net3, n_out=out_lst[3], k_szie=3, stride=2, pad='valid')

        return concatenate([net1, net2, net3], axis=-1)

    def _inception_resnetB(self,layer_id, net):

        net1 = self._conv_bn_relu(net=net, n_out=192, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b1_1')

        net2 = self._conv_bn_relu(net=net, n_out=128, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b2_1')
        net2 = self._conv_bn_relu(net=net2, n_out=160, k_szie=[1, 7], stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b2_2')
        net2 = self._conv_bn_relu(net=net2, n_out=192, k_szie=[7, 1], stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b2_3')

        net3 = concatenate([net1, net2], axis=-1)

        net3 = self._conv_bn(net=net3, n_out=1024, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetB_{layer_id}_b3_1')

        net4 = Add()([net3, net])
        return Activation('relu')(net4)

    def _reductionB(self, net):
        net1 = MaxPooling2D([3, 3], strides=[2, 2], padding='valid',name='reductionB_b1')(net)

        net2 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same',layer_name='reductionB_b2_1')
        net2 = self._conv_bn_relu(net=net2, n_out=384, k_szie=3, stride=2, pad='valid',layer_name='reductionB_b2_2')

        net3 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same',layer_name='reductionB_b3_1')
        net3 = self._conv_bn_relu(net=net3, n_out=256, k_szie=3, stride=2, pad='valid',layer_name='reductionB_b3_2')

        net4 = self._conv_bn_relu(net=net, n_out=256, k_szie=1, stride=1, pad='same',layer_name='reductionB_b4_1')
        net4 = self._conv_bn_relu(net=net4, n_out=256, k_szie=3, stride=1, pad='same',layer_name='reductionB_b4_2')
        net4 = self._conv_bn_relu(net=net4, n_out=256, k_szie=3, stride=2, pad='valid',layer_name='reductionB_b4_3')

        net = concatenate([net4, net1, net2, net3], axis=-1)
        return net

    def _inception_resnetC(self, layer_id,net):

        net1 = self._conv_bn_relu(net=net, n_out=192, k_szie=1, stride=1, pad='same',
                                  layer_name=f'inception_resnetC_{layer_id}_b1')

        net2 = self._conv_bn_relu(net=net, n_out=192, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetC_{layer_id}_b2_1')
        net2 = self._conv_bn_relu(net=net2, n_out=224, k_szie=[1, 3], stride=1, pad='same',layer_name=f'inception_resnetC_{layer_id}_b2_2')
        net2 = self._conv_bn_relu(net=net2, n_out=256, k_szie=[3, 1], stride=1, pad='same',layer_name=f'inception_resnetC_{layer_id}_b2_3')

        net3 = concatenate([net2, net1], axis=-1)
        net3 = self._conv_bn(net=net3, n_out=1920, k_szie=1, stride=1, pad='same',layer_name=f'inception_resnetC_{layer_id}_b3')
        net4 = Add()([net3, net])
        return Activation('relu')(net4)

    def build_inception_resnet_a(self, net, n_itsrs):
        for i in range(n_itsrs):
            self._inception_resnetA(layer_id=i,net=net)
        return net

    def build_inception_resnet_b(self, net, n_itsrs):
        for i in range(n_itsrs):
            self._inception_resnetB(layer_id=i,net=net)
        return net

    def build_inception_resnet_c(self, net, n_itsrs):
        for i in range(n_itsrs):
            self._inception_resnetC(layer_id=i,net=net)
        return net

    def build(self):
        data = Input(batch_shape=self.data_shape)

        net = self._stem(data)

        print('_stem  ', net.get_shape().as_list())
        net = self.build_inception_resnet_a(net=net, n_itsrs=5)
        print('build_inception_resnet_a ',net.get_shape().as_list())
        net = self._reductionA(net=net, out_lst=[384, 256, 384, 256])
        print('_reductionA  ',net.get_shape().as_list())

        net = self.build_inception_resnet_b(net, 10)
        print('build_inception_resnet_b  ',net.get_shape().as_list())
        net = self._reductionB(net)
        print('_reductionB  ', net.get_shape().as_list())

        net = self.build_inception_resnet_c(net, 5)
        print('_reductionC  ', net.get_shape().as_list())

        net = AveragePooling2D([8, 8], strides=[1, 1], padding='valid')(net)
        print('AveragePooling2D  ', net.get_shape().as_list())
        net = Dropout(0.8)(net)
        net = Flatten()(net)
        net = Dense(self.n_classes, activation='softmax')(net)
        print('Dense  ', net.get_shape().as_list())
        model = Model(inputs=data, outputs=net, name=self.model)
        model.summary()
        return net


inception_resnet_v2().build()
