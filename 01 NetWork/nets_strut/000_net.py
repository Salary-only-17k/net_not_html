from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D,AvgPool2D
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model
import keras as k
import numpy as np

"""
测试三个东西
1 全连接
2 卷积代替全连接
3 全局池化代替全连接
"""



def build_nets(flags=1):
    data_input = Input(batch_shape=(32,224,224,3))
    # -----1
    net = Conv2D(64,(3,3),strides=(1,1),padding="same",
                 activation="relu",name="conv1_1")(data_input)
    net = Conv2D(64,(3,3),strides=(1,1),padding="same",
                 activation="relu",name="conv1_2")(net)
    # -----2
    net = MaxPool2D((2,2),strides=(2,2),padding="Valid",name="maxpool1")(net)
    net = Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv2_1")(net)
    net = Conv2D(128, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv2_2")(net)
    # -----3
    net = MaxPool2D((2, 2), strides=(2, 2), padding="Valid", name="maxpool2")(net)
    net = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv3_1")(net)
    net = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv3_2")(net)
    net = Conv2D(256, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv3_3")(net)
    # -----4
    net = MaxPool2D((2, 2), strides=(2, 2), padding="Valid", name="maxpool3")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv4_1")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv4_2")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv4_3")(net)
    # -----5
    net = MaxPool2D((2, 2), strides=(2, 2), padding="Valid", name="maxpool4")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv5_1")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv5_2")(net)
    net = Conv2D(512, (3, 3), strides=(1, 1), padding="same",
                 activation="relu", name="conv5_3")(net)
    # ------6
    # conv2d  代替 fc
    net = MaxPool2D((2, 2), strides=(2, 2), padding="Valid", name="maxpool5")(net)

    if flags==1:
        # 普通的fc
        net = Flatten()(net)
        net = Dense(4096,name="fc1")(net)
        net = Activation("relu")(net)
        net = Dense(4096,name="fc2")(net)
        net = Activation("relu")(net)
        net = Dense(1000,name="fc3")(net)
        net = Activation("softmax")(net)
        a=net.get_layer("fc2").get_weights()
        print('a    --->   ',a)
        model = Model(inputs=data_input, outputs=net)
        print("output data shape : ", net.get_shape())
        return model
    elif flags==2:
        #  全局平均池化
        net = AvgPool2D((7,7),(1,1),padding="valid",name="pool_fc")(net)
        net = GlobalAveragePooling2D(name="GAP")(net)
        net = Dense(1000,name="fc1")(net)
        net = Activation("softmax")(net)
        model = Model(inputs=data_input, outputs=net)
        print("output data shape : ", net.get_shape())
        return model
    elif flags==3:
        # 卷积代替fc
        net = Conv2D(4096, (7, 7), strides=(1, 1), padding="valid",
                     activation="relu", name="cnn_fc1")(net)
        net = Conv2D(4096, (1,1), strides=(1, 1), padding="valid",
                     activation="relu", name="cnn_fc2")(net)
        net = Conv2D(1000, (1, 1), strides=(1, 1), padding="valid",
                     activation="relu", name="cnn_fc3")(net)
        net = Flatten()(net)
        net = Activation("softmax")(net)
        model = Model(inputs=data_input, outputs=net)
        print("output data shape : ",net.get_shape())
        return model
    else:
        raise ValueError

model = build_nets(2)
model.summary()