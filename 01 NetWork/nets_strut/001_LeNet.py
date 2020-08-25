from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model
import keras.activations as activation
'''
原味Lenet，没有bn，dropout之类的处理
2020229
chenglong
'''

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
        # net = BatchNormalization()
        net = MaxPool2D((2,2),strides=(2,2),name="s2")(net)
        net = Conv2D(16,(5,5),strides=(1,1),
                     activation="relu",padding="same",
                     kernel_regularizer=regularizers.l2(self.decay_weight),
                     name="c3")(net)
        net = MaxPool2D((2,2),strides=(2,2),name="s4")(net)
        net = Flatten()(net)
        net = Dense(120,name="f5",activation="sigmoid")(net)
        net = Dense(84,activation="sigmoid",name="f6")(net)
        net = Dense(self.n_classes, activation="softmax")(net)
        net = Activation('softmax')(net)
        model = Model(inputs=data_input,outputs=net)
        model.summary()
        from keras.utils import plot_model
        import os
        plot_model(model, to_file=os.path.join('./imgs', "001_Lent.png"), show_shapes=True)
        return net


model = Lenet()
model.build()













