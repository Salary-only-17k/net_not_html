from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate
from keras import regularizers
from keras.models import Model
import keras
import numpy as np


class inception_v4():
    def __init__(self, batch_size=32,
                 n_classes=1000):
        self.data_size = (batch_size, 299, 299, 3)
        self.n_classes = n_classes
        self.decay_weight = 0.5e-3

    def block_inception_a(self, layer_name, net, out_lst=[96,64,96,64,96,96,96]):
        # [96,64,96,64,96,96,96]
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 7, ValueError

        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv2_1')(net)
        net1 = Conv2D(out_lst[2], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv2_2')(net1)

        net2 = Conv2D(out_lst[3], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv3_1')(net)
        net2 = Conv2D(out_lst[4], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv3_2')(net2)
        net2 = Conv2D(out_lst[4], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv3_3')(net2)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + "avgpool4_1")(net)
        net3 = Conv2D(out_lst[5], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv4_2')(net3)

        return concatenate([net0, net1, net2, net3], axis=3)

    def block_reduction_a(self, layer_name, net, out_lst=[384,192,224,256]):
        # [384,192,224,256]
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 4, ValueError
        '''
                                         net
            net0-cnn-3x3-2      net1-max-3x3-2      net2-cnn-1x1-1
                                                    net2-cnn-3x3-1
                                                    net2-cnn-3x3-2
                                contract net0-net1-net2
        '''
        net0 = Conv2D(out_lst[0], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_1a_1x1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1 = Conv2D(out_lst[2], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0b_3x3')(net1)
        net1 = Conv2D(out_lst[3], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_1a_1x1')(net1)

        net2 = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name=layer_name + 'MaxPool2_1a_3x3')(net)
        return concatenate([net0, net1, net2], axis=3)

    def block_inception_b(self, layer_name, net, out_lst=[384,192,224,256,192,192,224,224,256,128]):
        # [384,192,224,256,192,192,224,224,256,128]
        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_0a_1x1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1 = Conv2D(out_lst[2], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0b_3x3')(net1)
        net1 = Conv2D(out_lst[3], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0c_5x5')(net1)

        net2 = Conv2D(out_lst[4], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0a_1x1')(net)
        net2 = Conv2D(out_lst[5], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0b_7x1')(net2)
        net2 = Conv2D(out_lst[6], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0c_7x1')(net2)
        net2 = Conv2D(out_lst[7], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0d_7x1')(net2)
        net2 = Conv2D(out_lst[8], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0e_1x7')(net2)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + 'AvgPool3_0a_3x3')(net)
        net3 = Conv2D(out_lst[9], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d3_0b_1x1')(net3)

        return concatenate([net0, net1, net2, net3], axis=3)

    def block_reduction_b(self, layer_name, net, out_lst=[192,192,256,256,620,620]):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 6, ValueError
        # [192,192,256,256,620,620]
        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_0a_1x1')(net)
        net0 = Conv2D(out_lst[0], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_1a_3x3')(net0)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1 = Conv2D(out_lst[2], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0b_1x7')(net1)
        net1 = Conv2D(out_lst[2], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0c_7x1')(net1)
        net1 = Conv2D(out_lst[1], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_1a_3x3')(net1)

        net2 = MaxPool2D((3, 3), strides=(2, 2), padding='valid',
                         name=layer_name + 'MaxPool2_1a_3x3')(net)
        return concatenate([net0, net1, net2], axis=3)

    def block_inception_c(self, layer_name, net, out_lst=[256,284,256,256,384,448,512,256,256,256]):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 9, ValueError

        # [256,284,256,256,384,448,512,256,256,256]
        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_0a_1x1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1_1 = Conv2D(out_lst[2], (1, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d1_1_0b_1x3')(net1)
        net1_2 = Conv2D(out_lst[3], (3, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d1_2_0c_3x1')(net1)
        net1 = concatenate([net1_1, net1_2], axis=3)

        net2 = Conv2D(out_lst[4], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0a_1x1')(net)
        net2 = Conv2D(out_lst[6], (3, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0b_3x1')(net2)
        net2 = Conv2D(out_lst[7], (1, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0c_1x3')(net2)
        net2_1 = Conv2D(out_lst[6], (1, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_1_0d_1x3')(net2)
        net2_2 = Conv2D(out_lst[7], (3, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_2_0e_3x1')(net2)
        net2 = concatenate([net2_1, net2_2], axis=3)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + 'AvgPool3_0a_3x3')(net)
        net3 = Conv2D(out_lst[8], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d3_0b_1x1')(net3)
        return concatenate([net0, net1, net2, net3], axis=3)

    def build(self):
        input_data = Input(batch_shape=self.data_size)
        net = Conv2D(32, [3, 3], strides=[2, 2], padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_1a_3x3')(input_data)
        net = Conv2D(32, [3, 3], strides=[1, 1], padding='valid', activation='relu',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_2a_3x3')(net)

        net = Conv2D(64, [3, 3], strides=[1, 1], padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_2b_3x3')(net)

        branch_net1_1 = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name='b1_1_maxPool')(net)
        branch_net1_2 = Conv2D(96, [3, 3], strides=[2, 2], padding='valid', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b1_1_Conv2d')(net)
        net = concatenate([branch_net1_1, branch_net1_2], axis=3)

        branch_net2_1_1 = Conv2D(96, [1, 1], strides=[2, 2], padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b2_1_1_Conv2d')(net)
        branch_net2_1_2 = Conv2D(64, [1, 7], strides=[2, 2], padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b2_1_2_Conv2d')(branch_net2_1_1)
        branch_net2_1_3 = Conv2D(64, [7, 1], strides=[2, 2], padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b2_1_3_Conv2d')(branch_net2_1_2)
        branch_net2_1_4 = Conv2D(96, [3, 3], strides=[2, 2], padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b2_1_4_Conv2d')(branch_net2_1_3)

        branch_net2_2_1 = Conv2D(64, [3, 3], strides=[1, 1], padding='same', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b1_2_1_Conv2d')(net)
        branch_net2_2_2 = Conv2D(96, [3, 3], strides=[1, 1], padding='valid', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b1_2_2_Conv2d')(branch_net2_2_1)
        net = concatenate([branch_net2_2_2, branch_net2_1_4], axis=3)

        branch_net3_1 = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name='b1_1_maxPool')(net)
        branch_net3_2 = Conv2D(192, [3, 3], strides=[2, 2], padding='valid', activation='relu',
                               kernel_regularizer=regularizers.l2(self.decay_weight),
                               name='b1_1_Conv2d')(net)
        net = concatenate([branch_net3_1, branch_net3_2], axis=3)

        for idn in range(4):
            layer_name = 'Mixed_5_' + str(idn)
            net = self.block_inception_a(layer_name,net)
        net = self.block_reduction_a('Mixed_6a',net)
        for idn in range(7):
            layer_name = 'Mixed_6_' + str(idn)
            net = self.block_inception_b(layer_name, net)
        net = self.block_reduction_a('Mixed_7a', net)
        for idn in range(3):
            layer_name = 'Mixed_7_' + str(idn)
            net = self.block_inception_c(layer_name, net)
        model = Model(inpust=input_data,outputs=net)
        from keras.utils import plot_model
        import os
        plot_model(model, to_file=os.path.join('./imgs', "007_inceptionv4.png"), show_shapes=True)
        model.summary()
        return model
    @property
    def input_size(self):
        return self.data_size[1:]
m = inception_v4()
m.build()