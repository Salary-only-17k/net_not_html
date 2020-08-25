from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model

from keras.layers.core import Layer

from keras import backend as K

'''
2020229
chenglong
'''


class LRN(Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha

        self.k = k

        self.beta = beta

        self.n = n

        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape

        half_n = self.n // 2  # half the local region

        # orig keras code

        # input_sqr = T.sqr(x)  # square the input

        input_sqr = K.square(x)  # square the input

        # orig keras code

        # extra_channels = T.alloc(0., b, ch + 2 * half_n, r,c)  # make an empty tensor with zero pads along channel dimension

        # input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input

        extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))

        input_sqr = K.concatenate(
            [extra_channels[:, :half_n, :, :], input_sqr, extra_channels[:, half_n + int(ch):, :, :]], axis=1)

        scale = self.k  # offset for the scale

        norm_alpha = self.alpha / self.n  # normalized alpha

        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]

        scale = scale ** self.beta

        x = x / scale

        return x

    def get_config(self):
        config = {"alpha": self.alpha,

                  "k": self.k,

                  "beta": self.beta,

                  "n": self.n}

        base_config = super(LRN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
class PoolHelper(Layer):

    def __init__(self, **kwargs):
        super(PoolHelper, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[:, :, 1:, 1:]

    def get_config(self):
        config = {}

        base_config = super(PoolHelper, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

class AlexNet():
    def __init__(self, data_size=(100, 227, 227, 3),
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
        data_input = Input(batch_shape=self.data_size)
        net = Conv2D(48,(11,11),strides=(4,4),
                     padding="valid",activation="relu",name="conv_1")(data_input)
        net = MaxPool2D((3,3),strides=(2,2),padding="valid",name="maxpool_1")(net)
        net = BatchNormalization()(net)

        net = Conv2D(128,(5,5),strides=(1,1),
                     padding="valid",activation="relu",name="conv_2")(net)
        net = MaxPool2D((3,3),strides=(2,2),padding="valid",name="maxpool_2")(net)
        net = BatchNormalization()(net)

        net = Conv2D(192,(3,3),strides=(1,1),
                     padding="valid",activation="relu",name="conv_3")(net)

        net = Conv2D(192, (3, 3), strides=(1, 1),
                     padding="valid", activation="relu", name="conv_4")(net)

        net = Conv2D(128, (3, 3), strides=(1, 1),
                     padding="valid", activation="relu", name="conv_5")(net)
        net = MaxPool2D((3,3),strides=(2,2),padding="valid",name="maxpool_3")(net)

        net = Flatten()(net)
        net = Dense(2048,activation="relu",name='fc6')(net)
        net = Dropout(0.5)(net)
        net = Dense(2048,activation="relu",name="fc7")(net)
        net = Dropout(0.5)(net)
        net = Dense(self.n_classes,activation="softmax")(net)

        model = Model(inputs=data_input, outputs=net)
        from keras.utils import plot_model
        import os
        plot_model(model, to_file=os.path.join('./imgs', "002_aLexnt.png"), show_shapes=True)
        model.summary()
        return net

model = AlexNet()
model.build()











