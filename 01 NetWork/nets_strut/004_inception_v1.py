from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate
from keras import regularizers
from keras.models import Model
import keras.activations as activation
import numpy as np

'''
2020301
chenglong
'''

class inception_V1():
    def __init__(self, batch_size=32,
                 n_classes=1000):

        self.data_size = (batch_size, 224, 224, 3)

        self.n_classes = n_classes
        self.decay_weight = 0.5e-3
        self.layer_lst = ['Conv2d_0a_1x1',
                          'Conv2d_0a_1x1', 'Conv2d_0b_3x3',
                          'Conv2d_0a_1x1', 'Conv2d_0b_3x3'
            , 'MaxPool_0a_3x3', 'Conv2d_0b_1x1']
    @property
    def input_size(self):
        return self.data_size[1:]

    def block(self, layer_name, net, out_lst):
        assert isinstance(out_lst, list), TypeError
        assert len(out_lst) == 6, ValueError

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
        net2 = Conv2D(out_lst[4], (5, 5), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv3_2')(net2)

        net3 = MaxPool2D((3, 3), strides=(1, 1),padding='same', name=layer_name +"maxpool3_1")(net)
        net3 = Conv2D(out_lst[5], (1, 1), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(self.decay_weight),
                      name=layer_name + 'conv5_2')(net3)

        return concatenate([net0, net1, net2, net3], axis=3)

    def subbranch(self, layer_name,net):
        net = AvgPool2D((5, 5), strides=(3, 3), padding='valid', name=layer_name + 'avgpool')(net)
        net = Conv2D(128, (1, 1), strides=(1, 1), padding='same',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name=layer_name + 'conv1')(net)
        net = Flatten(name=layer_name + 'Flatten1')(net)
        net = Dense(1024, activation='relu', name=layer_name + 'fc1')(net)
        net = Dropout(0.7,name=layer_name + 'dropout1')(net)
        net = Dense(self.n_classes, activation='relu', name=layer_name + 'fc2')(net)
        return Activation('softmax',name=layer_name + 'softmax')(net)


    def build(self):
        data_input = Input(batch_shape=self.data_size)
        net = Conv2D(64, (7, 7), strides=(2, 2),
                     padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name="conv1")(data_input)
        net = MaxPool2D((3, 3), strides=(2, 2),padding='same', name="maxpool1")(net)
        net = BatchNormalization()(net)

        net =  Conv2D(192, (1, 1), strides=(1, 1), padding='valid',
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name='conv2_1')(net)
        net = Conv2D(192, (3, 3), strides=(1, 1),
                     activation="relu", padding="same",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name="conv2_2")(net)
        net = BatchNormalization()(net)
        net = MaxPool2D((2, 2), strides=(2, 2),padding='same', name="maxpool2")(net)

        net = self.block('inception_3a_', net, [64, 96, 128, 16, 32, 32])
        net = self.block('inception_3b_', net, [128, 128, 192, 32, 96, 64])

        net = MaxPool2D((3, 3), strides=(2, 2),padding='same', name="maxpool3")(net)

        net = self.block('inception_4a_', net, [192, 96, 208, 16, 48, 64])
        loss01 = self.subbranch('sub_loss3_1_',net)
        net = self.block('inception_4b_', net, [160, 112, 224, 24, 64, 64])
        net = self.block('inception_4c_', net, [128, 128, 256, 24, 64, 64])
        net = self.block('inception_4d_', net, [112, 144, 288, 32, 64, 64])
        loss02 = self.subbranch('sub_loss3_2_', net)
        net = self.block('inception_4e_', net, [256, 160, 320, 32, 128, 128])

        net = MaxPool2D((3, 3), strides=(2, 2),padding='same', name="maxpool4")(net)

        net = self.block('inception_5a_', net, [256, 160, 320, 32, 128, 128])
        net = self.block('inception_5b_', net, [384, 192, 384, 48, 128, 128])

        net = AvgPool2D((7, 7), strides=(1, 1), padding='valid', name='avgpool3')(net)
        net = Dropout(0.4)(net)
        net = Flatten()(net)
        net = Dense(self.n_classes, name="fc1", activation="relu")(net)
        loss03 = Activation('softmax',name='loss3_3')(net)
        loss = 0.3*(loss01+loss02)+0.7*loss03
        model = Model(inputs=data_input, outputs=[loss01,loss02,loss03])
        from keras.utils import plot_model
        import os
        plot_model(model, to_file=os.path.join('./imgs', "004_inceptionv1.png"), show_shapes=True)
        model.summary()
        return loss


model = inception_V1()
model.build()
print(model.input_size)
