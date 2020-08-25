import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D,multiply,Add
from keras.layers import Dropout, BatchNormalization, Activation, concatenate, AvgPool2D, Lambda,DepthwiseConv2D
from keras.layers import ReLU
from keras import regularizers
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
SMALL_NET = [
    [16, 16, 2, True, 'relu', 3],
    [72, 24, 2, False, 'relu', 3],
    [88, 24, 1, False, 'relu', 3],
    [96, 40, 2, True, 'swich', 5],
    [240, 40, 1, True, 'swich', 5],
    [240, 40, 1, True, 'swich', 5],
    [120, 48, 1, True, 'swich', 5],
    [114, 48, 1, True, 'swich', 5],
    [288, 96, 2, True, 'swich', 5],
    [576, 96, 1, True, 'swich', 5],
    [576, 96, 1, True, 'swich', 5],
]

BIG_NET = [
    [16, 16, 1, False, 'relu', 3],
    [64, 24, 2, False, 'relu', 3],
    [72, 24, 1, False, 'relu', 3],
    [72, 40, 2, True, 'relu', 5],
    [120, 40, 1, True, 'relu', 5],
    [120, 40, 1, True, 'relu', 3],
    [240, 80, 2, False, 'swich', 3],
    [200, 80, 1, False, 'swich', 3],
    [184, 80, 1, False, 'swich', 3],
    [184, 80, 1, False, 'swich', 3],
    [480,112, 1, True, 'swich', 3],
    [672, 112, 1, True, 'swich', 5],
    [672, 160, 2, True, 'swich', 5],
    [960, 160, 1, True, 'swich', 5],
    [960, 160, 1, True, 'swich', 5],
]

class mobilenet_v3():
    def __init__(self, batch_size, n_classes, flags='small', action='relu'):
        # flags : 'small'  'big' 我就不想用large 字太多了
        # action
        self.model_name = 'se-net'
        self.data_size = [batch_size, 224, 224, 3]
        self.n_classes = n_classes
        self.decay_weight = 0.5e-3
        if flags.lower()=='small':
            self.params=SMALL_NET
        else:
            self.params=BIG_NET
        if action == 'relu':
            func = self._swish_relu6
        elif action == 'sigmid':
            func = self._swish_sigmoid
        else:
            raise ValueError
        get_custom_objects().update({'swish': Activation(func)})

    def _swish_relu6(self, value):
        return K.relu(value, max_value=6.0) * value

    def _swish_sigmoid(self, value):
        return K.sigmoid(value) * value

    def _SE_block(self, net, r=16):
        residual = net
        i_in = net.get_shape().as_list()[-1]
        net = GlobalAveragePooling2D()(net)
        net = Dense(i_in // r, activation='relu')(net)
        net = Dense(i_in)(net)
        net = Activation('swish')(net)
        net = Lambda(lambda x: K.expand_dims(x, axis=1))(net)
        net = Lambda(lambda x: K.expand_dims(x, axis=1))(net)
        net = multiply([residual,net])
        return net
    def _bottle_neck(self,net, exp_size, out_channels, dwstride, is_se_block, AC, dw_ksize):
        # AC  "relu"  'swich'
        AC = AC.lower()
        residual = net
        in_channels = net.get_shape().as_list()[-1]
        net = Conv2D(exp_size,[1,1],strides=[1,1],padding='same')(net)
        net =BatchNormalization()(net)
        if AC=='relu':
            net = Activation('relu')(net)
        else:
            net = Activation('swish')(net)
        net = DepthwiseConv2D([dw_ksize,dw_ksize],strides=[dwstride, dwstride],padding='same')(net)
        net = BatchNormalization()(net)
        if AC=='relu':
            net = Activation('relu')(net)
        else:
            net = Activation('swish')(net)
        if is_se_block:
            net = self._SE_block(net)
        net = Conv2D(out_channels,[1,1],strides=[1,1],padding='same')(net)
        net = BatchNormalization()(net)
        import keras
        net = keras.activations.linear(net)
        if dwstride == 1 and in_channels == out_channels:
            net = Add()([residual, net])
        return net
    @property
    def img_size(self):
        return self.data_size[1:]
    def build(self):
        data = Input(batch_shape=self.data_size)
        net = Conv2D(16,[3,3],strides=[2,2],padding='same')(data)
        net = BatchNormalization()(net)
        for param in self.params:
            net = self._bottle_neck(net,*param)
        n_in = self.params[-1][1]
        net = Conv2D(n_in,[1,1],strides=[1,1],padding='same')(net)
        net = BatchNormalization()(net)
        net = GlobalAveragePooling2D()(net)
        net = Dense(1028)(net)
        net = Dense(self.n_classes,activation='relu')(net)

        # 这里用cnn代替fc，但是这个惭怍需要把2维数据添加到4威，我不想写代码了，直接用fc了
        # net = Conv2D(1028,[1,1],strides=[1,1],padding='same')(net)
        # net = Conv2D(self.n_classes,[1,1],strides=[1,1],padding='same',activation='softmax')(net)

        model = Model(inputs=data,outputs=net,name=self.model_name)
        model.summary()
        return net

mobilenet_v3(32,1000,flags='big', action='relu').build()