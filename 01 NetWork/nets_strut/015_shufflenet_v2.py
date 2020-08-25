import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D, concatenate
import keras as k


class shufflenet_v1():
    def __init__(self, include_top=True, scale_factor=1.0, pooling='max', batch_size=32, groups=1,
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=0.25, n_classes=1000):
        '''
        :param include_top:  是否味stay的顶层
        :param scale_factor: 调整输出通道的数量
        :param pooling:   全局池化的方式
        :param batch_size:  batch大小
        :param groups:   论文中的g
        :param num_shuffle_units:   stay中 shuffle_units重复次数
        :param bottleneck_ratio:     0.25=1/4  论文中瓶颈层 in_c/out_c =1/4
        :param n_classes:  类
        '''
        self.name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (
            scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
        self.data_shape = [batch_size, 224, 224, 3]
        out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 224}
        '''
        stay2  out_dim_stage_two[i]
        stay3  out_dim_stage_two[i] * 2
        stay4  out_dim_stage_two[i] * 4
        下面是一个可扩展长度的代码  可变长度
        exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
        out_channels_in_stage = 2 ** exp
        out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
        out_channels_in_stage[0] = 24  # first stage has always 24 output channels
        out_channels_in_stage *= scale_factor
        self.out_channels_in_stage = out_channels_in_stage.astype(int)
        '''
        value = out_dim_stage_two[groups]
        self.out_channels_in_stage = [24, value, value * 2, value * 4]
        self.n_classes = n_classes
        self.num_shuffle_units = num_shuffle_units
        self.pooling = pooling
        self.groups = groups
        self.bottleneck_ratio = bottleneck_ratio
        self.include_top = include_top

    @property
    def img_size(self):
        return self.data_shape[1:]

    def _stage(self, net, out_channels, bottleneck_ratio, repeat=1, groups=1, stage=1):
        '''
                        1
                        net
                net1           1x1 gcnov
                               bn relu
                               channel shuffle
                               3x3 DWconv1
                               bn
                               1x1 gconv  bn
                        add

                        2-n
                        net
                avg            1x1 gcnov
                               bn relu
                               channel shuffle
                               3x3 DWconv2
                               bn
                               1x1 gconv
                               bn
                        concat
        :param net:  model
        :param channel_map:
        :param bottleneck_ratio: 1/4
        :param repeat:
        :param groups:
        :param stage:
        :return:
        '''
        net = self._shuffle_unit(net,out_channels=out_channels[stage - 1], strides=2
                                 , bottleneck_ratio=bottleneck_ratio,
                                 stage=stage, block=1)

        for i in range(1, repeat + 1):
            net = self._shuffle_unit(net,
                                     out_channels=out_channels[stage - 1], strides=1,
                                     bottleneck_ratio=bottleneck_ratio,
                                     stage=stage, block=(i+1))

        return net

    def _shuffle_unit(self, net, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):
        """
        creates a shuffleunit

        Parameters
        ----------
        inputs:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        strides:
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
        groups: int(1)
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
            For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
            the width of the bottleneck feature map.
        stage: int(1)
            stage number
        block: int(1)
            block number

        Returns
        -------

        """
        if K.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        in_channels = net.get_shape().as_list()[3]
        prefix = 'stage%d/block%d' % (stage, block)
        bottleneck_channels = int(out_channels * bottleneck_ratio)
        if strides == 1:
            c, c_hat = self.channel_split('{}/spl'.format(prefix), net)
            net = c

            net = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same',
                         name='{}/1x1conv_1'.format(prefix))(net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(net)
            net = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(net)
            net = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv_1s'.format(prefix))(
                net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_1s'.format(prefix))(net)
            net = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same',
                         name='{}/1x1conv_2'.format(prefix))(net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(net)
            net = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(net)

            net = concatenate([net, c_hat], axis=3)
            net = Lambda(self.channel_shuffle, name='{}/channel_shuffle'.format(prefix))(net)
            return net
        elif strides == 2:
            residual = net

            net = Conv2D(bottleneck_channels, kernel_size=(1, 1), strides=1, padding='same',
                         name='{}/1x1conv_1'.format(prefix))(net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(net)
            net = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(net)
            net = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv_2s'.format(prefix))(
                net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2s'.format(prefix))(net)
            net = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same',
                         name='{}/1x1conv_2_1'.format(prefix))(net)
            net = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2_1'.format(prefix))(net)
            net = Activation('relu', name='{}/relu_1x1conv_2_1'.format(prefix))(net)

            residual = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same',
                                       name='{}/3x3dwconv'.format(prefix))(
                residual)
            residual = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(residual)
            residual = Conv2D(bottleneck_channels, kernel_size=1, strides=1, padding='same',
                              name='{}/1x1conv_2_2'.format(prefix))(residual)
            residual = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2_2'.format(prefix))(residual)
            residual = Activation('relu', name='{}/relu_1x1conv_2_2'.format(prefix))(residual)
            net = concatenate([net, residual], axis=3)
            net = Lambda(self.channel_shuffle, name='{}/channel_shuffle'.format(prefix))(net)
            return net
        else:
            assert ValueError, "strides not 1 and not 2  is error"

    def channel_split(self, layer_name, net):
        in_channel = net.get_shape().as_list()[3]
        sp = in_channel // 2
        c = Lambda(lambda x: x[:, :, :, 0:sp], name='%s/sp%d_slice' % (layer_name, 1))(net)
        c_hat = Lambda(lambda x: x[:, :, :, sp:], name='%s/sp%d_slice' % (layer_name, 0))(net)
        return c, c_hat

    def channel_shuffle(self, tensor, groups=2):
        # print('groups==self.groups',groups==self.groups)

        height, width, in_channels = tensor.shape.as_list()[1:]
        channels_per_group = in_channels // groups
        tensor = K.reshape(tensor, [-1, height, width, groups, channels_per_group])
        tensor = K.permute_dimensions(tensor, (0, 1, 2, 4, 3))  # transpose
        tensor = K.reshape(tensor, [-1, height, width, in_channels])
        return tensor

    def build(self):

        # create shufflenet architecture
        data_size = Input(batch_shape=self.data_shape)
        net = Conv2D(filters=self.out_channels_in_stage[0], kernel_size=[3, 3], strides=[2, 2], padding='same',
                     use_bias=False, activation="relu", name="conv1")(data_size)
        net = MaxPooling2D([3, 3], strides=[2, 2], padding='same', name="maxpool1")(net)

        # create stages containing shufflenet units beginning at stage 2
        for stage in range(0, len(self.num_shuffle_units)):
            repeat = self.num_shuffle_units[stage]  # 每个阶段 对应的重复次数
            net = self._stage(net, self.out_channels_in_stage, repeat=repeat,
                              bottleneck_ratio=self.bottleneck_ratio,
                              groups=self.groups, stage=stage + 2)

        if self.bottleneck_ratio < 2:
            n_out = 1024
        else:
            n_out = 2048
        net = Conv2D(n_out, [1, 1], strides=[1, 1], padding='same', name='1x1conv5_out', activation='relu')(net)
        if self.pooling == 'avg':
            net = GlobalAveragePooling2D(name="global_pool")(net)
        elif self.pooling == 'max':
            net = GlobalMaxPooling2D(name="global_pool")(net)

        net = Dense(units=self.n_classes, name="fc")(net)
        net = Activation('softmax', name='softmax')(net)
        model = Model(inputs=data_size, outputs=net, name=self.name)
        model.summary()

        return model


shufflenet_v1(groups=1).build()
