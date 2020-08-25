import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation, concatenate,multiply,Lambda
from keras import regularizers
from keras.models import Model
import keras.activations as activation
import keras.backend as K
import keras
class se_resnet_x():
    def __init__(self, batch_size, n_classes,n_block_lst):
        self.model_name = 'se-net'
        self.data_size = [batch_size, 224, 224, 3]
        self.n_classes = n_classes
        self.decay_weight = 0.5e-3
        self.n_block_lst=n_block_lst

    @property
    def img_size(self):
        return self.data_size[1:]

    def _se_block(self,net,r=16):
        n_in = net.get_shape().as_list()[-1]
        residual = net
        net = GlobalAveragePooling2D()(net)
        net = Dense(n_in//r,activation='relu')(net)
        net = Dense(n_in,activation='sigmoid')(net)
        net = Lambda(lambda x:K.expand_dims(x,axis=1))(net)
        net = Lambda(lambda x:K.expand_dims(x,axis=1))(net)
        net = multiply(inputs=[net,residual])
        return net

    def _BottleNeck(self,n_out,net,stride):
        net = Conv2D(n_out*4,[1,1],strides=[stride,stride],padding='valid')(net)
        net = BatchNormalization()(net)

        net =Conv2D(n_out,[1,1],strides=[1,1],padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        net = Conv2D(n_out,[3,3],strides=[stride,stride],padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        net = Conv2D(n_out*4, [1, 1], strides=[1,1], padding='same')(net)
        net = BatchNormalization()(net)
        net = self._se_block(net)
        net = Activation('relu')(net)
        return net
    def _make_res_block(self,n_out,net,n_blocks,stride=1):
        net = self._BottleNeck(n_out,net=net,stride=stride)
        for _ in range(1,n_blocks):
            net = self._BottleNeck(n_out,net=net,stride=1)
        return net
    def build(self):
        data = Input(batch_shape=self.data_size)
        net = Conv2D(64,[7,7],strides=[2,2],padding='same')(data)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net=MaxPool2D([3,3],strides=[2,2])(net)
        net = self._make_res_block(n_out=64, net=net, n_blocks=self.n_block_lst[0])
        net = self._make_res_block(n_out=128, net=net, n_blocks=self.n_block_lst[1],stride=2)
        net = self._make_res_block(n_out=256, net=net, n_blocks=self.n_block_lst[2],stride=2)
        net = self._make_res_block(n_out=512, net=net, n_blocks=self.n_block_lst[3],stride=2)
        net = GlobalAveragePooling2D()(net)
        net = Dense(self.n_classes,activation='softmax')(net)
        model = Model(inputs=data,outputs=net,name=self.model_name)
        model.summary()
        return net
    @staticmethod
    def se_resnet_50(batch_size,n_classes):
        return se_resnet_x(batch_size,n_classes,n_block_lst=[3, 4, 6, 3]).build()

    @staticmethod
    def se_resnet_101(batch_size,n_classes):
        return se_resnet_x(batch_size,n_classes,[3, 4, 23, 3]).build()

    @staticmethod
    def se_resnet_152(batch_size,n_classes):
        return se_resnet_x(batch_size,n_classes,[3, 8, 36, 3]).build()



se_resnet_x.se_resnet_50(32,1000)