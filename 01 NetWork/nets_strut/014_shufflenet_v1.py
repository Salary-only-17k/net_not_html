import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda
from keras.layers import DepthwiseConv2D
import numpy as np


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
        out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
        '''
        stay2  out_dim_stage_two[i]
        stay3  out_dim_stage_two[i] * 2
        stay4  out_dim_stage_two[i] * 4
        下面是一个可扩展长度的代码
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
        net = self._shuffle_unit(net,
                                 out_channels=out_channels[stage - 1], strides=2,
                                 groups=groups, bottleneck_ratio=bottleneck_ratio,
                                 stage=stage, block=1)

        for i in range(1, repeat + 1):
            net = self._shuffle_unit(net,
                                     out_channels=out_channels[stage - 1], strides=1,
                                     groups=groups, bottleneck_ratio=bottleneck_ratio,
                                     stage=stage, block=(i + 1))

        return net

    def _shuffle_unit(self, net,out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
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
        residual = net
        in_channels = net.get_shape().as_list()[3]
        prefix = 'stage%d/block%d' % (stage, block)

        # if strides >= 2:
        # out_channels -= in_channels

        # default: 1/4 of the output channel of a ShuffleNet Unit
        # 默认值：ShuffleNet单元输出通道的1/4
        bottleneck_channels = int(out_channels * bottleneck_ratio)  # 算出瓶颈输出通道
        groups = (1 if stage == 2 and block == 1 else groups) # 判断是不是1阶段或者第2stay

        net = self._group_conv(net, out_channels=bottleneck_channels,
                               groups=(1 if stage == 2 and block == 1 else groups),
                               name='%s/1x1_gconv_1' % prefix)
        net = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(net)
        net = Activation('relu', name='%s/relu_gconv_1' % prefix)(net)

        net = Lambda(self.channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(net)
        net = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
                              strides=strides, name='%s/1x1_dwconv_1' % prefix)(net)
        net = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(net)

        net = self._group_conv(net, out_channels=out_channels if strides == 1 else out_channels - in_channels,
                               groups=groups, name='%s/1x1_gconv_2' % prefix)
        net = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(net)

        if strides < 2:
            ret = Add(name='%s/add' % prefix)([net, residual])
        else:
            residual = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(residual)
            ret = Concatenate(bn_axis, name='%s/concat' % prefix)([net, residual])

        net = Activation('relu', name='%s/relu_out' % prefix)(ret)

        return net

    def _group_conv(self, net, out_channels, groups, kernel=1, stride=1, name=''):
        # print('groups==self.groups', groups == self.groups)
        """
        grouped convolution
        Parameters
        ----------
        net:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        groups:
            number of groups per channel
        kernel: int(1)
            An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        stride: int(1)
            An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for all spatial dimensions.
        name: str
            A string to specifies the layer name

        Returns
        -------

        """
        if groups == 1:
            return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                          use_bias=False, strides=stride, name=name)(net)

        # number of intput channels per group
        in_channels = net.get_shape().as_list()[3]
        ig = in_channels // groups
        group_list = []

        assert out_channels % groups == 0

        for i in range(groups):
            offset = i * ig
            group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(net)
            group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                     use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
        return Concatenate(name='%s/concat' % name)(group_list)

    def channel_shuffle(self, tensor, groups):
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
        net = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool1")(net)

        # create stages containing shufflenet units beginning at stage 2
        for stage in range(0, len(self.num_shuffle_units)):
            repeat = self.num_shuffle_units[stage]  # 每个阶段 对应的重复次数
            net = self._stage(net, self.out_channels_in_stage, repeat=repeat,
                             bottleneck_ratio=self.bottleneck_ratio,
                             groups=self.groups, stage=stage + 2)

        if self.pooling == 'avg':
            net = GlobalAveragePooling2D(name="global_pool")(net)
        elif self.pooling == 'max':
            net = GlobalMaxPooling2D(name="global_pool")(net)

        # if self.include_top:
        #     net = Dense(units=self.n_classes, name="fc")(net)
        #     net = Activation('softmax', name='softmax')(net)
        net = Dense(units=self.n_classes, name="fc")(net)
        net = Activation('softmax', name='softmax')(net)

        model = Model(inputs=data_size, outputs=net, name=self.name)
        model.summary()

        return model


# class ShuffleNet():
#     """
#     ShuffleNet implementation for Keras 2
#
#     ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
#     Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
#     https://arxiv.org/pdf/1707.01083.pdf
#
#     Note that only TensorFlow is supported for now, therefore it only works
#     with the data format `image_data_format='channels_last'` in your Keras
#     config at `~/.keras/keras.json`.
#
#     Parameters
#     ----------
#     include_top: bool(True)
#          whether to include the fully-connected layer at the top of the network.
#     input_tensor:
#         optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
#     scale_factor:
#         scales the number of output channels
#     input_shape:
#     pooling:
#         Optional pooling mode for feature extraction
#         when `include_top` is `False`.
#         - `None` means that the output of the model
#             will be the 4D tensor output of the
#             last convolutional layer.
#         - `avg` means that global average pooling
#             will be applied to the output of the
#             last convolutional layer, and thus
#             the output of the model will be a
#             2D tensor.
#         - `max` means that global max pooling will
#             be applied.
#     groups: int
#         number of groups per channel
#     num_shuffle_units: list([3,7,3])
#         number of stages (list length) and the number of shufflenet units in a
#         stage beginning with stage 2 because stage 1 is fixed
#
#         e.g. idx 0 contains 3 + 1 (first shuffle unit in each stage differs) shufflenet units for stage 2
#         idx 1 contains 7 + 1 Shufflenet Units for stage 3 and
#         idx 2 contains 3 + 1 Shufflenet Units
#     bottleneck_ratio:
#         bottleneck ratio implies the ratio of bottleneck channels to output channels.
#         For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
#         the width of the bottleneck feature map.
#     classes: int(1000)
#         number of classes to predict
#     Returns
#     -------
#         A Keras model instance
#
#     References
#     ----------
#     - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices]
#       (http://www.arxiv.org/pdf/1707.01083.pdf)
#
#     """
#
#     def __init__(self, include_top=True, scale_factor=1.0, pooling='max', batch_size=32, groups=1,
#                  num_shuffle_units=[3, 7, 3],
#                  bottleneck_ratio=0.25, n_classes=1000):
#         self.name = "ShuffleNet_%.2gX_g%d_br_%.2g_%s" % (
#             scale_factor, groups, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
#         self.data_shape = [batch_size, 224, 224, 3]
#         out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
#
#         exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
#         out_channels_in_stage = 2 ** exp
#         out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
#         out_channels_in_stage[0] = 24  # first stage has always 24 output channels
#         out_channels_in_stage *= scale_factor
#         self.out_channels_in_stage = out_channels_in_stage.astype(int)
#         self.n_classes = n_classes
#         self.num_shuffle_units = num_shuffle_units
#         self.pooling = pooling
#         self.groups = groups
#         self.bottleneck_ratio = bottleneck_ratio
#         self.include_top = include_top
#
#     def _block(self, net, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
#         """
#         creates a bottleneck block containing `repeat + 1` shuffle units
#
#         Parameters
#         ----------
#         x:
#             Input tensor of with `channels_last` data format
#         channel_map: list
#             list containing the number of output channels for a stage
#         repeat: int(1)
#             number of repetitions for a shuffle unit with stride 1
#         groups: int(1)
#             number of groups per channel
#         bottleneck_ratio: float
#             bottleneck ratio implies the ratio of bottleneck channels to output channels.
#             For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
#             the width of the bottleneck feature map.
#         stage: int(1)
#             stage number
#
#         Returns
#         -------
#
#         """
#         net = self._shuffle_unit(net, in_channels=channel_map[stage - 2],
#                                  out_channels=channel_map[stage - 1], strides=2,
#                                  groups=groups, bottleneck_ratio=bottleneck_ratio,
#                                  stage=stage, block=1)
#
#         for i in range(1, repeat + 1):
#             net = self._shuffle_unit(net, in_channels=channel_map[stage - 1],
#                                      out_channels=channel_map[stage - 1], strides=1,
#                                      groups=groups, bottleneck_ratio=bottleneck_ratio,
#                                      stage=stage, block=(i + 1))
#
#         return net
#
#     def _shuffle_unit(self, inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
#         """
#         creates a shuffleunit
#
#         Parameters
#         ----------
#         inputs:
#             Input tensor of with `channels_last` data format
#         in_channels:
#             number of input channels
#         out_channels:
#             number of output channels
#         strides:
#             An integer or tuple/list of 2 integers,
#             specifying the strides of the convolution along the width and height.
#         groups: int(1)
#             number of groups per channel
#         bottleneck_ratio: float
#             bottleneck ratio implies the ratio of bottleneck channels to output channels.
#             For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
#             the width of the bottleneck feature map.
#         stage: int(1)
#             stage number
#         block: int(1)
#             block number
#
#         Returns
#         -------
#
#         """
#         if K.image_data_format() == 'channels_last':
#             bn_axis = -1
#         else:
#             bn_axis = 1
#
#         prefix = 'stage%d/block%d' % (stage, block)
#
#         # if strides >= 2:
#         # out_channels -= in_channels
#
#         # default: 1/4 of the output channel of a ShuffleNet Unit
#         # 默认值：ShuffleNet单元输出通道的1/4
#         bottleneck_channels = int(out_channels * bottleneck_ratio)
#         groups = (1 if stage == 2 and block == 1 else groups)
#
#         net = self._group_conv(inputs, in_channels, out_channels=bottleneck_channels,
#                                groups=(1 if stage == 2 and block == 1 else groups),
#                                name='%s/1x1_gconv_1' % prefix)
#         net = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(net)
#         net = Activation('relu', name='%s/relu_gconv_1' % prefix)(net)
#
#         net = Lambda(self.channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(net)
#         net = DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=False,
#                               strides=strides, name='%s/1x1_dwconv_1' % prefix)(net)
#         net = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(net)
#
#         net = self._group_conv(net, bottleneck_channels,
#                                out_channels=out_channels if strides == 1 else out_channels - in_channels,
#                                groups=groups, name='%s/1x1_gconv_2' % prefix)
#         net = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(net)
#
#         if strides < 2:
#             ret = Add(name='%s/add' % prefix)([net, inputs])
#         else:
#             avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
#             ret = Concatenate(bn_axis, name='%s/concat' % prefix)([net, avg])
#
#         ret = Activation('relu', name='%s/relu_out' % prefix)(ret)
#
#         return ret
#
#     def _group_conv(self, x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
#         """
#         grouped convolution
#         Parameters
#         ----------
#         x:
#             Input tensor of with `channels_last` data format
#         in_channels:
#             number of input channels
#         out_channels:
#             number of output channels
#         groups:
#             number of groups per channel
#         kernel: int(1)
#             An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#             Can be a single integer to specify the same value for
#             all spatial dimensions.
#         stride: int(1)
#             An integer or tuple/list of 2 integers,
#             specifying the strides of the convolution along the width and height.
#             Can be a single integer to specify the same value for all spatial dimensions.
#         name: str
#             A string to specifies the layer name
#
#         Returns
#         -------
#
#         """
#         if groups == 1:
#             return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
#                           use_bias=False, strides=stride, name=name)(x)
#
#         # number of intput channels per group
#         ig = in_channels // groups
#         group_list = []
#
#         assert out_channels % groups == 0
#
#         for i in range(groups):
#             offset = i * ig
#             group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
#             group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
#                                      use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
#         return Concatenate(name='%s/concat' % name)(group_list)
#
#     def channel_shuffle(self, x, groups):
#         """
#         Parameters
#         ----------
#         x:
#             Input tensor of with `channels_last` data format
#         groups: int
#             number of groups per channel
#
#         Returns
#         -------
#             channel shuffled output tensor
#
#         Examples
#         """
#         height, width, in_channels = x.shape.as_list()[1:]
#         channels_per_group = in_channels // groups
#         x = K.reshape(x, [-1, height, width, groups, channels_per_group])
#         x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
#         x = K.reshape(x, [-1, height, width, in_channels])
#         return x
#
#     def build(self):
#
#         # create shufflenet architecture
#         data_size = Input(batch_shape=self.data_shape)
#         net = Conv2D(filters=self.out_channels_in_stage[0], kernel_size=(3, 3), padding='same',
#                      use_bias=False, strides=(2, 2), activation="relu", name="conv1")(data_size)
#         net = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name="maxpool1")(net)
#
#         # create stages containing shufflenet units beginning at stage 2
#         for stage in range(0, len(self.num_shuffle_units)):
#             repeat = self.num_shuffle_units[stage]
#             net = self._block(net, self.out_channels_in_stage, repeat=repeat,
#                               bottleneck_ratio=self.bottleneck_ratio,
#                               groups=self.groups, stage=stage + 2)
#
#         if self.pooling == 'avg':
#             net = GlobalAveragePooling2D(name="global_pool")(net)
#         elif self.pooling == 'max':
#             net = GlobalMaxPooling2D(name="global_pool")(net)
#
#         if self.include_top:
#             net = Dense(units=self.n_classes, name="fc")(net)
#             net = Activation('softmax', name='softmax')(net)
#
#         model = Model(inputs=data_size, outputs=net, name=self.name)
#         model.summary()
#
#         return model


shufflenet_v1(groups=1).build()
