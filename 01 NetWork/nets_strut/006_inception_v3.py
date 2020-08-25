from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate
from keras import regularizers
from keras.models import Model
import keras
import numpy as np


class inception_V3():
    def __init__(self, batch_size=32,
                 n_classes=1000):
        self.data_size = (batch_size, 299, 299, 3)
        self.n_classes = n_classes
        self.decay_weight = 0.5e-3

    @property
    def input_size(self):
        return self.data_size[1:]

    def block_5(self, layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 7, ValueError
        '''
                                          net
            net0-cnn-1x1-1      net1-cnn-1x1-1      net2-cnn-1x1-1       net3-avg-3x3-1
                                net1-cnn-5x5-1      net2-cnn-3x3-1       net3-cnn-1x1-1
                                                    net2-cnn-3x3-1
                                contract net0-net1-net2-net3
        '''
        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_0a_1x1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        # net1 = Conv2D(out_lst[2], (3, 3), strides=(1, 1), padding='same',
        #               kernel_regularizer=regularizers.l2(self.decay_weight),
        #               name=layer_name + 'Conv2d_0b_3x3')(net1)
        net1 = Conv2D(out_lst[2], (5, 5), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0b_5x5')(net1)

        net2 = Conv2D(out_lst[3], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0a_1x1')(net)
        net2 = Conv2D(out_lst[4], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0b_3x3')(net2)
        net2 = Conv2D(out_lst[5], (3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0c_3x3')(net2)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + 'AvgPool3_0a_3x3')(net)
        net3 = Conv2D(out_lst[6], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d3_0b_1x1')(net3)

        return concatenate([net0, net1, net2, net3], axis=3)

    def block_6(self, layer_name, net, out_lst):
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

    def block_7(self, layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 10, ValueError
        '''
                                         net
            net0-cnn-1x1-1      net1-cnn-1x1-1      net2-cnn-1x1-1       net3-avg-3x3-1
                                net1-cnn-1x7-1      net2-cnn-7x1-1       net3-cnn-1x1-1
                                net1-cnn-7x1-1      net2-cnn-1x7-1
                                                    net2-cnn-7x1-1
                                                    net2-cnn-1x7-1
                                                    
                                contract net0-net1-net2-net3
        '''
        net0 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d0_0a_1x1')(net)

        net1 = Conv2D(out_lst[1], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1 = Conv2D(out_lst[2], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0b_1x7')(net1)
        net1 = Conv2D(out_lst[3], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0c_7x1')(net1)

        net2 = Conv2D(out_lst[4], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0a_1x1')(net)
        net2 = Conv2D(out_lst[5], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0b_7x1')(net2)
        net2 = Conv2D(out_lst[6], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0c_1x7')(net2)
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

    def block_8(self, layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 6, ValueError
        '''
                                     net
            net0-cnn-1x1-2      net1-max-3x3-2      net2-cnn-3x3-1
            net0-cnn-3x3-2                          net2-cnn-1x7-1
                                                    net2-cnn-7x1-1
                                                    net2-cnn-3x3-2
                                contract net0-net1-net2
        '''
        net1 = Conv2D(out_lst[0], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_0a_1x1')(net)
        net1 = Conv2D(out_lst[1], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d1_1a_3x3')(net1)

        net2 = Conv2D(out_lst[2], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0a_1x1')(net)
        net2 = Conv2D(out_lst[3], (1, 7), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0b_1x7')(net2)
        net2 = Conv2D(out_lst[4], (7, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_0c_7x1')(net2)
        net2 = Conv2D(out_lst[5], (3, 3), strides=(2, 2), padding='valid',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d2_1a_3x3')(net2)

        net3 = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name=layer_name + "MaxPool3_1a_3x3")(net)
        return concatenate([net1, net2, net3], axis=3)

    def block_9(self, layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 9, ValueError
        '''
                                        net
            net0-cnn-1x1-1      net1-cnn-1x1-1              net2-cnn-1x1-1            net3-avg-3x3-1  
                                net1_1-cnn-1x3-1            net2-cnn-3x3-1            net3-cnn-1x1-1
                                net1_2-cnn-3x1-1            net2_1-cnn-1x3-1            
                                contract net1_1-net1_2      net2_2-cnn-3x1-1              
                                                            contract net2_1-net2_2
                                            contract net0-net1-net2 
        '''
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
        net2 = Conv2D(out_lst[5], (3, 3), strides=(1, 1), padding='same',
                             kernel_regularizer=regularizers.l2(self.decay_weight),
                             name=layer_name + 'Conv2d2_0b_3x3')(net2)
        net2_1 = Conv2D(out_lst[6], (1, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_1_0c_1x3')(net2)
        net2_2 = Conv2D(out_lst[7], (3, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_2_0d_3x1')(net2)
        net2 = concatenate([net2_1, net2_2], axis=3)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + 'AvgPool3_0a_3x3')(net)
        net3 = Conv2D(out_lst[8], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d3_0b_1x1')(net3)
        return concatenate([net0, net1, net2, net3], axis=3)
    def block_10(self,layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 8, ValueError
        '''
                                        net
            net0-cnn-1x1-1      net1-cnn-1x1-1              net2-cnn-1x1-1            net3-avg-3x3-1  
                                net1_1-cnn-1x3-1            net2-cnn-3x3-1            net3-cnn-1x1-1
                                net1_2-cnn-3x1-1            net2_1-cnn-1x3-1            
                                contract net1_1-net1_2      net2_2-cnn-3x1-1              
                                                            contract net2_1-net2_2
                                            contract net0-net1-net2 
        '''
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
        net2_1 = Conv2D(out_lst[5], (1, 3), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_1_0b_1x3')(net2)
        net2_2 = Conv2D(out_lst[6], (3, 1), strides=(1, 1), padding='same',
                        kernel_regularizer=regularizers.l2(self.decay_weight),
                        name=layer_name + 'Conv2d2_2_0c_3x1')(net2)
        net2 = concatenate([net2_1, net2_2], axis=3)

        net3 = AvgPool2D((3, 3), strides=(1, 1), padding='same', name=layer_name + 'AvgPool3_0a_3x3')(net)
        net3 = Conv2D(out_lst[7], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'Conv2d3_0b_1x1')(net3)
        return concatenate([net0, net1, net2, net3], axis=3)

    def subbranch(self, layer_name, net):
        net = AvgPool2D((5, 5), strides=(3, 3), padding='valid', name=layer_name + 'AvgPool_1a_5x5')(net)
        net = Conv2D(128, (1, 1), strides=(1, 1), padding='same',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name=layer_name + 'Conv2d_1b_1x1')(net)
        kernel_size =self._reduced_kernel_size_for_small_input(net,[5,5])
        net = Conv2D(768, kernel_size, strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name=layer_name + 'Conv2c_1c_1x1')(net)
        net = Conv2D(self.n_classes, [1,1], strides=(1, 1), padding='same',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name=layer_name + 'Conv2d_1d_1x1')(net)
        net = Flatten(name=layer_name + 'Flatten1')(net)
        # net = Dropout(0.7, name=layer_name + 'dropout1')(net)
        return Activation('softmax', name=layer_name + 'softmax')(net)

    def build(self):
        '''
        下面结构来源于slim
        Here is a mapping from the old_names to the new names:
        Old name          | New name
        =======================================
        conv0             | Conv2d_1a_3x3
        conv1             | Conv2d_2a_3x3
        conv2             | Conv2d_2b_3x3
        pool1             | MaxPool_3a_3x3
        conv3             | Conv2d_3b_1x1
        conv4             | Conv2d_4a_3x3
        pool2             | MaxPool_5a_3x3.

        mixed_35x35x256a  | Mixed_5b
        mixed_35x35x288a  | Mixed_5c
        mixed_35x35x288b  | Mixed_5d

        mixed_17x17x768a  | Mixed_6a
        mixed_17x17x768b  | Mixed_6b
        mixed_17x17x768c  | Mixed_6c
        mixed_17x17x768d  | Mixed_6d
        mixed_17x17x768e  | Mixed_6e

        mixed_8x8x1280a   | Mixed_7a
        mixed_8x8x2048a   | Mixed_7b
        mixed_8x8x2048b   | Mixed_7c
        '''
        data_input = Input(batch_shape=self.data_size)
        net = Conv2D(32, (3, 3), strides=(2, 2),
                     padding="valid", activation="relu",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name="Conv2d_1a_3x3")(data_input)
        net = Conv2D(32, (3, 3), strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_2a_3x3')(net)
        net = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_2b_3x3')(net)
        net = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name="MaxPool_3a_3x3")(net)

        net = Conv2D(80, (1, 1), strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_3b_1x1')(net)
        net = Conv2D(192, (3, 3), strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_4a_3x3')(net)
        net = MaxPool2D((3, 3), strides=(2, 2), padding='valid', name="MaxPool_5a_3x3")(net)

        net = self.block_5('Mixed_5b_', net, [64, 48, 64, 64, 96, 96, 32])
        net = self.block_5('Mixed_5c_', net, [64, 48, 64, 64, 96, 96, 64])
        net = self.block_5('Mixed_5d_', net, [64, 48, 64, 64, 96, 96, 64])

        net = self.block_6('Mixed_6a_', net, [384, 64, 96, 96])
        # loss01 = self.subbranch('sub_loss3_1_', net)
        net = self.block_7('Mixed_6b_', net, [192, 128, 128, 192, 128, 128, 128, 128, 192, 192])
        net = self.block_7('Mixed_6c_', net, [192, 160, 160, 192, 160, 160, 160, 160, 192, 192])
        net = self.block_7('Mixed_6d_', net, [192, 160, 160, 192, 160, 160, 160, 160, 192, 192])
        net = self.block_7('Mixed_6e_', net, [192, 192, 192, 192, 192, 192, 192, 192, 192, 192])
        loss01 = self.subbranch('sub_loss3_2_', net)

        net = self.block_8('Mixed_7a_', net, [192, 320, 192, 192, 192, 192])

        net = self.block_9('Mixed_7b', net, [320, 384, 384, 384, 448, 384,384,384,192])
        net = self.block_10('Mixed_7c_', net, [320, 384, 384, 384, 448, 384, 384, 192])

        net = Dropout(0.8,name='Dropout_1b')(net)
        net = Conv2D(self.n_classes, (8, 8), strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='Conv2d_1c_1x1')(net)

        net = Flatten()(net)
        loss02 = Activation('softmax', name='loss3_3')(net)
        loss = 0.3 * loss01 + 0.7 * loss02
        model = Model(inputs=data_input, outputs=[loss01, loss02])
        from keras.utils import plot_model
        import os
        plot_model(model, to_file=os.path.join('./imgs', "006_inceptionv3.png"), show_shapes=True)
        model.summary()
        return loss

    # @staticmethod
    def _reduced_kernel_size_for_small_input(self,input_tensor, kernel_size):

        shape = input_tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            kernel_size_out = kernel_size
        else:
            kernel_size_out = [min(shape[1], kernel_size[0]),
                               min(shape[2], kernel_size[1])]
        return kernel_size_out


model = inception_V3()
model.build()
