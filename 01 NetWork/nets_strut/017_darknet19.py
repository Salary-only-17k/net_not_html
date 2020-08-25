import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate, Lambda
from keras import regularizers
from keras.models import Model
import keras.backend as K
from keras.layers.merge import add as Kadd
import keras.activations as activation


class darkent19():
    def __init__(self, batch_size=32, n_classes=1000, weight_decay=0.0004, flags=None):
        self.data_size = [batch_size, 416, 416, 3]
        self.n_classes = n_classes
        self.weight_decay = weight_decay

    @property
    def input_size(self):
        return self.data_size[1:]

    def _conv(self, layer_name, net, n_out, ks, sd, pad='same'):
        '''
        :param layer_name:
        :param net:
        :param n_out:
        :param ks:kernel size
        :param sd: stride
        :param pad:
        :return:
        '''
        return Conv2D(n_out, [ks, ks], strides=[sd, sd],
                      kernel_regularizer=regularizers.l2(self.weight_decay),
                      # ernel_initializer="he_normal",
                      activation='relu',
                      padding=pad, name=layer_name)(net)

    def _maxpool(self, layer_name, net, ks, sd, pad='valid'):
        return MaxPool2D([ks, ks], strides=[sd, sd], padding=pad, name=layer_name)(net)

    def _reorg_layer(self,net):
        def expand_dims(x):
            return K.expand_dims(x, axis=3)

        data_shape = net.get_shape().as_list()
        r, channels = int(data_shape[1] / 2), data_shape[3]
        tensor = []
        for i in range(channels):
            ur, ul, dr, dl = net[:, :r, :r, i], net[:, :r, r:, i], net[:, r:, :r, i], net[:, r:, r:, i]
            ur = expand_dims(ur)
            tensor.append(ur)
            ul = expand_dims(ul)
            tensor.append(ul)
            dr = expand_dims(dr)
            tensor.append(dr)
            dl = expand_dims(dl)
            tensor.append(dl)
        return concatenate(tensor, axis=3)

    def build(self):
        data = Input(batch_shape=self.data_size)
        net = self._conv('conv0', data, 32, 3, 1, 'same')
        net = self._maxpool('maxpool1', net, 2, 2, 'valid')

        net = self._conv('conv2', net, 64, 3, 1, 'same')
        net = self._maxpool('maxpool3', net, 2, 2, 'valid')

        net = self._conv('conv4', net, 128, 3, 1, 'same')
        net = self._conv('conv5', net, 64, 1, 1, 'same')
        net = self._conv('conv6', net, 128, 3, 1, 'same')
        net = self._maxpool('maxpool7', net, 2, 2, 'valid')

        net = self._conv('conv8', net, 64, 3, 1, 'same')
        net = self._conv('conv9', net, 64, 3, 1, 'same')
        net = self._conv('conv10', net, 64, 3, 1, 'same')
        net = self._maxpool('maxpool11', net, 2, 2, 'valid')

        net = self._conv('conv12', net, 512, 3, 1, 'same')
        net = self._conv('conv13', net, 256, 1, 1, 'same')
        net = self._conv('conv14', net, 512, 3, 1, 'same')
        net = self._conv('conv15', net, 256, 1, 1, 'same')
        net1_0 = self._conv('conv16', net, 512, 3, 1, 'same')
        net = self._maxpool('maxpool17', net1_0, 2, 2, 'valid')

        net = self._conv('conv18', net, 1024, 3, 1, 'same')
        net = self._conv('conv19', net, 512, 1, 1, 'same')
        net = self._conv('conv20', net, 1024, 3, 1, 'same')
        net = self._conv('conv21', net, 512, 1, 1, 'same')
        net = self._conv('conv22', net, 1024, 3, 1, 'same')
        net = self._conv('conv23', net, 1024, 3, 1, 'same')
        net1_1 = self._conv('conv24', net, 1024, 3, 1, 'same')
        net = net1_0
        net2_0 = Lambda(self._reorg_layer, name='reorg_layer')(net)  # 26
        net = concatenate([net2_0, net1_1], axis=3)  # 27
        net = self._conv('conv28', net, 1024, 3, 1, 'same')
        net = self._conv('conv29', net, self.n_classes, 1, 1, 'same')
        net = Activation('softmax')(net)
        model = Model(inputs=data, outputs=net)
        model.summary()
        return model


darkent19().build()
print(darkent19().data_size)
