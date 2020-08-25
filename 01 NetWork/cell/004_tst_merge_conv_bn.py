import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model
import keras.backend as K


class Lenet():
    def __init__(self,data_size=(100,32,32,3),
                 n_classes=10):
        # 5
        if len(data_size) == 4:
            self.data_size = data_size
        else:
            raise TypeError
        self.n_classes = n_classes
        self.decay_weight = 0.5e-3
    @property
    def data_shape(self):
        return self.data_size[1:]
    def build(self):
        def mergeConvBn():
            def inner(net):
                beta = 1E-5
                axis = list(range(len(net.get_shape().as_list())))
                mean, var = K.mean(net, axis), K.var(net, axis)
                alpha = 1 / K.sqrt(var + beta)
                return alpha, mean, var
            return inner

        def weight_init(shape, dtype=None):
            nonlocal alpha, mean
            return K.random_normal(shape, dtype=dtype) * alpha

        def bias_init(shape=1, dtype=None):
            nonlocal alpha, mean
            return alpha * K.ones(shape, dtype=dtype) + mean / alpha
        data_input=Input(batch_shape=self.data_size)

        alpha, mean, var = mergeConvBn()(data_input)
        net =Conv2D(6,(5,5),strides=(1,1),
                    padding="valid",activation="sigmoid",
                    kernel_regularizer=regularizers.l2(self.decay_weight),
                    kernel_initializer=weight_init,
                    bias_initializer=bias_init,
                    name="c1")(data_input)
        net = MaxPool2D((2,2),strides=(2,2),name="s2")(net)

        alpha, mean, var = mergeConvBn()(net)
        net = Conv2D(16,(5,5),strides=(1,1),
                     activation="relu",padding="same",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     kernel_initializer=weight_init,
                     bias_initializer=bias_init,
                     name="c3")(net)
        net = MaxPool2D((2,2),strides=(2,2),name="s4")(net)

        net = Flatten()(net)
        net = Dense(120,name="f5",activation="sigmoid")(net)
        net = Dense(84,activation="sigmoid",name="f6")(net)
        net = Dense(self.n_classes, activation="softmax")(net)
        net = Activation('softmax')(net)
        model = Model(inputs=data_input,outputs=net)
        model.summary()
        return net


model = Lenet()
model.build()





