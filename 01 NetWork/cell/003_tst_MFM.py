import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation,maximum,concatenate,Lambda
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
        data_input=Input(batch_shape=self.data_size)

        net =Conv2D(6,(5,5),strides=(1,1),
                    padding="valid",activation="sigmoid",
                    kernel_regularizer=regularizers.l2(self.decay_weight),
                    name="c1")(data_input)
        net = Lambda(self.MFM,name='MFM1')(net)

        net = Conv2D(16,(5,5),strides=(1,1),
                     activation="relu",padding="same",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name="c3")(net)
        net = Lambda(self.MFM,name='MFM2')(net)

        net = Flatten()(net)
        net = Dense(120,name="f5",activation="sigmoid")(net)
        net = Dense(84,activation="sigmoid",name="f6")(net)
        net = Dense(self.n_classes, activation="softmax")(net)
        net = Activation('softmax')(net)
        model = Model(inputs=data_input,outputs=net)
        model.summary()
        return net

    def MFM(self,net):
        n = int(net.get_shape().as_list()[3]/2)
        tmp = []
        for i in range(n):
            tmp_value = K.maximum(net[...,i],net[...,-i])
            tmp.append(K.expand_dims(tmp_value,axis=3))
        return concatenate(tmp,axis=3)



model = Lenet()
model.build()





