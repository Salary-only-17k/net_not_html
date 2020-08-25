import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.utils.data_utils import get_file
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'


class Xception():
    def __init__(self,batch_szie,n_classes=1000,weight_decay=0.00001,flags=1):
        self.data_shape = [batch_szie,299,299,3]
        self.n_classes = n_classes
        self.flags = flags
        self.weight_decay = weight_decay
    @property
    def img_size(self):
        return self.data_shape[1:]
    def build(self):
        data = Input(batch_shape=self.data_shape)
        # Block 1
        net = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(data)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = Conv2D(64, (3, 3), use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(net)
        residual = BatchNormalization()(residual)
        # Block 2
        net = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        # Block 2 Pool
        net = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)
        net = layers.add([net, residual])
        residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(net)
        residual = BatchNormalization()(residual)
        # Block 3
        net = Activation('relu')(net)
        net = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        # Block 3 Pool
        net = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)
        net = layers.add([net, residual])
        residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(net)
        residual = BatchNormalization()(residual)
        # Block 4
        net = Activation('relu')(net)
        net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)
        net = layers.add([net, residual])
        # Block 5 - 12
        for i in range(8):
            residual = net
            net = Activation('relu')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = Activation('relu')(net)
            net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
            net = BatchNormalization()(net)
            net = layers.add([net, residual])
        residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(net)
        residual = BatchNormalization()(residual)
        # Block 13
        net = Activation('relu')(net)
        net = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        # Block 13 Pool
        net = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net)
        net = layers.add([net, residual])
        # Block 14
        net = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        # Block 14 part 2
        net = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        # Fully Connected Layer
        net = GlobalAveragePooling2D()(net)
        net = Dense(1000, activation='softmax')(net)

        # Create model
        model = Model(inputs=data, outputs=net, name='xception')
        model.summary()
        # # Download and cache the Xception weights file
        # weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')
        # # load weights
        # model.load_weights(weights_path)
        return model


"""
Instantiate the model by using the following line of code
model = Xception()
"""
Xception(32).build()