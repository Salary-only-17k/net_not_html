from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Lambda, Add, Activation, concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, AvgPool2D
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
import keras.backend as K

'''
20200228
chenglong
03
20200229
chenglong
02
20200229
chenglong
031
'''


class ResNext():
    def __init__(self, batch_shape=32, cardinality=32, sub_block_id=1, include_top=True, n_classes=1000, rate=1):
        # self.data = [batch_shape, 32, 32, 3]
        self.data = [batch_shape, 224, 224, 3]
        self.weight_decay = 0.0001
        self.n_lasses = n_classes
        self.include_top = include_top
        self.depth_lst = [3, 4, 6, 3]
        self.cardinality = cardinality
        # filter = 8 * 32
        self.n_out_lst = {1: 128,
                          2: 256,
                          3: 512,
                          4: 1024}
        block_total = {1: self._bottleneck_block_01,
                       2: self._bottleneck_block_02,
                       3: self._bottleneck_block_03}
        self.sub_block = block_total[sub_block_id]
        # print(self.sub_block)
        self.rate = rate

    def build(self):
        data = Input(batch_shape=self.data)
        # n_out = self.cardinality * self.width
        net = Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False,
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(self.weight_decay),
                     name='conv1')(data)  # 这里修改了
        # print("conv1  0", net.get_shape().as_list())  # 对
        net = BatchNormalization()(net)
        net = Activation('relu')(net)

        net = MaxPool2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(net)

        for i in range(self.depth_lst[0]):
            net = self.sub_block(layer_name=f'sub_block1_{i}_', net=net, stride=1, n_out=self.n_out_lst[1],
                                 top_block=True)
        # for i in range(self.depth_lst[1]):
        #     if i == 0:
        #         net = self.sub_block(layer_name=f'sub_block2_{i}_', net=net, stride=2, n_out=self.n_out_lst[2])
        #     else:
        #         net = self.sub_block(layer_name=f'sub_block2_{i}_', net=net, stride=1, n_out=self.n_out_lst[2])
        # for i in range(self.depth_lst[2]):
        #     if i == 0:
        #         net = self.sub_block(layer_name=f'sub_block3_{i}_', net=net, stride=2, n_out=self.n_out_lst[3])
        #     else:
        #         net = self.sub_block(layer_name=f'sub_block3_{i}_', net=net, stride=1, n_out=self.n_out_lst[3])
        # for i in range(self.depth_lst[3]):
        #     if i == 0:
        #         net = self.sub_block(layer_name=f'sub_block4_{i}_', net=net, stride=2, n_out=self.n_out_lst[4])
        #     else:
        #         net = self.sub_block(layer_name=f'sub_block4_{i}_', net=net, stride=1, n_out=self.n_out_lst[4])
        net = GlobalAveragePooling2D()(net)
        net = Dense(self.n_lasses, name='logits')(net)
        model = Model(inputs=data, outputs=net, name='resnext')
        model.summary()
        return net

    def _bottleneck_block_01(self, layer_name, net, n_out, stride, top_block=False):
        residual = net
        assert n_out % self.cardinality == 0, 'n_out应该能被self.cardinality整除'
        split_channels = int(n_out / self.cardinality)
        if top_block:
            net = self._grouped_convolution_block(n_out=n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=3, depth=3)  # TODO

            print('residual 1 ', residual.get_shape().as_list())
            print("net  1  ", net.get_shape().as_list())
            net = Add()([residual, net])
            return Activation('relu')(net)

        else:

            net = self._grouped_convolution_block(n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=3, depth=1)

            print('residual  ', residual.get_shape().as_list())
            print("net  ", net.get_shape().as_list())
            net = Add()([residual, net])
            return Activation('relu')(net)

    def _bottleneck_block_02(self, layer_name, net, n_out, stride, top_block=False):

        residual = net
        assert n_out % self.cardinality == 0, 'n_out应该能被self.cardinality整除'
        split_channels = int(n_out / self.cardinality)
        if top_block:

            net = self._grouped_convolution_block(n_out=n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=3, depth=2)

            net = Conv2D(n_out * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay),
                         name=layer_name + "conv_3")(net)
            net = BatchNormalization()(net)
            print('residual 1 ', residual.get_shape().as_list())
            print("net  1  ", net.get_shape().as_list())
            net = Add()([residual, net])
            net = Activation('relu')(net)
            return net
        else:
            net = self._grouped_convolution_block(n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=3, depth=2)

            net = Conv2D(n_out * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay))(net)
            net = BatchNormalization()(net)
            print('residual  ', residual.get_shape().as_list())
            print("net  ", net.get_shape().as_list())
            net = Add()([residual, net])
            net = Activation('relu')(net)

            return net

    def _bottleneck_block_03(self, layer_name, net, n_out, stride=1, top_block=False):
        residual = net

        assert n_out % self.cardinality == 0, 'n_out应该能被self.cardinality整除'
        split_channels = int(n_out / self.cardinality)

        if top_block:
            net = Conv2D(n_out, (1, 1), strides=[1, 1], padding='same', use_bias=False,
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                         name=layer_name + "conv_1_1")(net)
            # print("net  0", net.get_shape().as_list())
            net = BatchNormalization()(net)
            net = Activation('relu')(net)

            net = self._grouped_convolution_block(n_out=n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=stride, depth=1)

            net = Conv2D(n_out * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay),
                         name=layer_name + "conv_1_2")(net)
            net = BatchNormalization()(net)

            if residual.get_shape().as_list()[-1] != n_out * 2:
                residual = Conv2D(n_out * 2, [1, 1], strides=[stride, stride],
                                  padding='same', use_bias=False,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(self.weight_decay),
                                  name=layer_name + "conv_3")(residual)
                residual = BatchNormalization()(residual)
            net = Add()([residual, net])
            net = Activation('relu')(net)
            return net
        else:
            net = Conv2D(n_out, (1, 1), strides=[stride, stride],
                         padding='same', use_bias=False,
                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay)
                         , name=layer_name + 'conv1')(net)

            net = BatchNormalization()(net)
            net = Activation('relu')(net)

            net = self._grouped_convolution_block(n_out,
                                                  layer_name=layer_name, net=net, split_channels=split_channels,
                                                  strides=1, depth=1)

            net = Conv2D(n_out * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
                         kernel_regularizer=l2(self.weight_decay))(net)

            net = BatchNormalization()(net)

            if residual.get_shape().as_list()[-1] != n_out * 2:
                residual = Conv2D(n_out * 2, (1, 1), strides=[stride, stride], padding='same', use_bias=False,
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(self.weight_decay),
                                  name=layer_name + "conv_3")(residual)

                # residual = MaxPool2D([2, 2], strides=[1, 1], padding='valid')(residual)

            net = Add()([residual, net])
            net = Activation('relu')(net)

            return net

    def _grouped_convolution_block(self, n_out, layer_name, net, split_channels, strides, depth):
        # merge_way=merge.lower()
        # assert merge_way in ['add','concatenate','none']
        group_list = []
        if depth == 1:
            # 论文中第三种方法
            if self.cardinality == 1:
                # with cardinality 1, it is a standard convolution
                net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
                             padding='same', use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(self.weight_decay), name=layer_name)(net)
                net = BatchNormalization()(net)
                net = Activation('relu')(net)
                return net
            else:
                for c in range(self.cardinality):
                    seplit_net = Lambda(lambda z: z[:, :, :, c * split_channels:(c + 1) * split_channels])(net)

                    seplit_net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
                                        padding='same', use_bias=False,
                                        kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                                        name=layer_name + f'_conv_{c}')(seplit_net)

                    group_list.append(seplit_net)
                group_merge = concatenate(group_list, axis=-1)

                net = BatchNormalization()(group_merge)
                net = Activation('relu')(net)
                return net
        elif depth == 2:
            # residual = net
            for c in range(self.cardinality):
                seplit_net = Lambda(lambda z: z[:, :, :, c * split_channels:(c + 1) * split_channels])(net)
                seplit_net = Conv2D(split_channels, (1, 1), strides=[strides, strides],
                                    padding='same', use_bias=False,
                                    kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                                    name=layer_name + f'_conv_{c}')(seplit_net)
                seplit_net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
                                    padding='same', use_bias=False,
                                    kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                                    name=layer_name + f'_conv_{c + 1}')(seplit_net)

                group_list.append(seplit_net)

            group_merge = concatenate(group_list, axis=-1)
            net = BatchNormalization()(group_merge)
            net = Activation('relu')(net)
            return net

        elif depth == 3:
            if self.cardinality == 1:
                # with cardinality 1, it is a standard convolution
                net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
                             padding='same', use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(self.weight_decay), name=layer_name)(net)
                net = BatchNormalization()(net)
                net = Activation('relu')(net)
                return net
            else:
                for c in range(self.cardinality):
                    seplit_net = Lambda(lambda z: z[:, :, :, c * split_channels:(c + 1) * split_channels])(net)

                    seplit_net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
                                        padding='same', use_bias=False,
                                        kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
                                        name=layer_name + f'_conv_{c}')(seplit_net)

                    group_list.append(seplit_net)
                group_merge = concatenate(group_list, axis=-1)

                net = BatchNormalization()(group_merge)
                net = Activation('relu')(net)
                return net
            # residual = net
            # for c in range(self.cardinality):
            #     seplit_net = Lambda(lambda z: z[:, :, :, c * split_channels:(c + 1) * split_channels])(net)
            #     seplit_net = Conv2D(split_channels, (1, 1), strides=[strides, strides],
            #                         padding='same', use_bias=False,
            #                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
            #                         name=layer_name + f'_conv_{c}')(seplit_net)
            #     seplit_net = Conv2D(split_channels, (3, 3), strides=[strides, strides],
            #                         padding='same', use_bias=False,
            #                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
            #                         name=layer_name + f'_conv_{c + 1}')(seplit_net)
            #     seplit_net = Conv2D(n_out, (1, 1), strides=[strides, strides],
            #                         padding='same', use_bias=False,
            #                         kernel_initializer='he_normal', kernel_regularizer=l2(self.weight_decay),
            #                         name=layer_name + f'_conv_{c + 2}')(seplit_net)
            #
            #     group_list.append(seplit_net)
            #
            # group_merge = Add()(group_list)
            # net = BatchNormalization()(group_merge)
            # net = Activation('relu')(net)
            # return Add()(net, residual)


if __name__ == '__main__':
    # (self, batch_shape=32, cardinality=32, sub_block_id=1,include_top=True, n_classes=10,rate=1)
    model = ResNext(batch_shape=32, sub_block_id=1).build()
