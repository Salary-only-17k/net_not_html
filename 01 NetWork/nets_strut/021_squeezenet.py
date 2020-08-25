import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate, AvgPool2D
from keras import regularizers
from keras.models import Model


class squeezenet():
    def __init__(self, batch, n_classes):
        self.model_name = 'se-net'
        self.data_size = [batch, 224, 224, 3]

        self.n_classes = n_classes
        self.decay_weight = 0.5e-3

    @property
    def img_size(self):
        return self.data_size[1:]

    def _FireModule(self, layer_name, net, n_out_lst):
        assert len(n_out_lst) == 3
        net = Conv2D(n_out_lst[0], [1, 1], strides=[1, 1], padding='same',
                     activation='relu', name=layer_name + 'squeeze1')(net)
        net1 = Conv2D(n_out_lst[1], [1, 1], strides=[1, 1], padding='same',
                      activation='relu', name=layer_name + 'conv1x1')(net)
        net2 = Conv2D(n_out_lst[2], [3, 3], strides=[1, 1], padding='same',
                      activation='relu', name=layer_name + 'conv3x3')(net)
        return concatenate([net1, net2], axis=-1)

    def build(self):
        data = Input(batch_shape=self.data_size)
        net = Conv2D(96, [7, 7], strides=[2, 2], padding='same', name='conv1')(data)
        net = MaxPool2D([3, 3], strides=[2, 2], padding='valid')(net)

        net = self._FireModule('FireM_2_', net, [16, 64, 64])
        net = self._FireModule('FireM_3_', net, [16, 64, 64])
        net = self._FireModule('FireM_4_', net, [32, 32, 128])
        net = MaxPool2D([3, 3], strides=[2, 2], padding='valid', name='maxpool4')(net)

        net = self._FireModule('FireM_5_', net, [32, 32, 128])
        net = self._FireModule('FireM_6_', net, [48, 48, 192])
        net = self._FireModule('FireM_7_', net, [48, 48, 192])
        net = self._FireModule('FireM_8_', net, [64, 256, 256])
        net = MaxPool2D([3, 3], strides=[2, 2], padding='valid', name='maxpool8')(net)

        net = self._FireModule('FireM_9_', net, [64, 256, 256])

        net = Dropout(0.5)(net)
        net = Conv2D(self.n_classes, [1, 1], strides=[1, 1], padding='same', name='conv10')(net)
        net = GlobalAveragePooling2D()(net)
        net = Activation('softmax')(net)
        model = Model(inputs=data, outputs=net, name=self.model_name)
        model.summary()
        return net


squeezenet(32, 1000).build()
