from keras.layers import \
    (Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D,AvgPool2D)
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model
import numpy as np

decay_weight=0.001

'''
2020229
chenglong
'''
class VGG_N:
    def __init__(self,
                 data_size=(32, 224, 224, 3),
                 n_classes=1000,

                 value_dtype=np.float16,
                 decay_weight=0.0005,
                 max_epoch=1600,

                 flags=1,

                 model_name="VGG16"):

        self.flags=flags

        if len(data_size) == 4:
            self.data_size = data_size
        else:
            raise TypeError
        self.n_classes = n_classes
        if model_name.lower() in ["vgg16", "vgg19"]:
            self.model_name = model_name
        else:
            raise NameError

        self.decay_weight = 0.0005
        self.max_epoch = 1600

    def build_VGG16(self):
        data_input = Input(batch_shape=self.data_size)  # ,dtype=np.float32)

        net = Conv2D(64, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding='same', activation="relu",
                     name="conv1_1")(data_input)  # 可以
        net = Conv2D(64, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding='same', activation="relu",
                     name="conv1_2")(net)
        net = MaxPool2D((2, 2), strides=(2, 2), name="maxpool_1")(net)

        net = Conv2D(128, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv2_1")(net)
        net = Conv2D(128, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv2_2")(net)
        net = MaxPool2D((2, 2), strides=(2, 2), name="maxpool_2")(net)

        net = Conv2D(256, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv3_1")(net)
        net = Conv2D(256, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv3_2")(net)
        net = Conv2D(256, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv3_3")(net)
        net = MaxPool2D((2, 2), strides=(2, 2), name="maxpool3")(net)

        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv4_1")(net)
        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv4_2")(net)
        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv4_3")(net)
        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv4_4")(net)
        net = MaxPool2D((2, 2), strides=(2, 2), name="maxpool4")(net)

        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv5_1")(net)
        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv5_2")(net)
        net = Conv2D(512, (3, 3), strides=(1, 1),
                     kernel_regularizer=regularizers.l2(decay_weight), padding="same", activation="relu",
                     name="conv5_3")(net)
        net = MaxPool2D((2, 2), strides=(2, 2), name="maxpool5")(net)

        # net = Flatten()(net) # 全局池化不可以的

        if self.flags==1:
            # 普通的fc
            net = Flatten()(net)
            net = Dense(4096, name="fc1")(net)
            net = Activation("relu")(net)
            net = Dense(4096, name="fc2")(net)
            net = Activation("relu")(net)
            net = Dense(self.n_classes, name="fc3")(net)
            net = Activation("softmax")(net)

            model = Model(inputs=data_input, outputs=net)
            from keras.utils import plot_model
            import os
            plot_model(model, to_file=os.path.join('./imgs', "003_vgg.png"), show_shapes=True)
            model.summary()
            print("output data shape : ", net.get_shape())
            return net
        elif self.flags== 2:
            #  全局平均池化
            net = AvgPool2D((7, 7), (1, 1), padding="valid", name="pool_fc")(net)
            net = GlobalAveragePooling2D(name="GAP")(net)
            net = Dense(self.n_classes, name="fc1")(net)
            net = Activation("softmax")(net)
            model = Model(inputs=data_input, outputs=net)
            model.summary()
            print("output data shape : ", net.get_shape())
            return net
        elif self.flags== 3:
            # 卷积代替fc
            fks = net.get_shape().as_list()[1:3]
            net = Conv2D(4096, fks, strides=(1, 1), padding="valid",
                         activation="relu", name="cnn_fc1")(net)
            net = Conv2D(self.n_classes, (1, 1), strides=(1, 1), padding="valid",
                         activation="relu", name="cnn_fc3")(net)
            net = Activation("softmax")(net)
            model = Model(inputs=data_input, outputs=net)
            print("output data shape : ", net.get_shape())

            model.summary()
            return net
        else:
            raise ValueError

    # def build_VGG19(self):
    #     pass
    #
    # def __repr__(self):
    #     self.model.summary()
    # def load_weight(self):
    #     self.build_VGG16.load

model = VGG_N()
model.build_VGG16()
