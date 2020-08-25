from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, Dense
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Lambda, Dropout, Add, multiply
from keras.layers import DepthwiseConv2D, concatenate
import keras as k
import tensorflow as tf
import math
from keras.utils.generic_utils import get_custom_objects

__all__ = ['EfficientNet',
           'EfficientNetB0',
           'EfficientNetB1',
           'EfficientNetB2',
           'EfficientNetB3',
           'EfficientNetB4',
           'EfficientNetB5',
           'EfficientNetB6',
           'EfficientNetB7',
           'preprocess_input']


class efficientnet():
    def __init__(self, width_coefficient,
                 depth_coefficient, dropout_rate,
                 batch_size=32, n_clases=1000,
                 drop_connect_rate=0.2,
                 action='sigmoid', ratio=0.25):
        self.data_size = [batch_size, 224, 224, 3]
        self.n_classes = n_clases
        self.ratio = ratio
        if action == 'relu':
            func = self._swish_relu6
        elif action == 'sigmoid':
            func = self._swish_sigmoid
        else:
            raise ValueError
        get_custom_objects().update({'swish': Activation(func)})
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate

    def _swish_relu6(self, value):
        return K.relu(value, max_value=6.0) * value

    def _swish_sigmoid(self, value):
        return K.sigmoid(value) * value

    def _round_filters(self, k_size, multiple, depth_divisor, min_depth):
        assert isinstance(multiple, float)
        assert isinstance(depth_divisor, int)
        if not multiple:
            return k_size

        min_depth = min_depth or depth_divisor
        filters = k_size * multiple
        new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
        if new_filters < 0.9 * filters:
            new_filters += depth_divisor
        return int(new_filters)

    def _round_repeat(self, repeats, multiple):
        if not multiple:
            return repeats
        else:
            return int(math.ceil(multiple * repeats))

    def drop_connect(self, net, drop_connect_rate):
        keep_prob = 1.0 - drop_connect_rate

        # Compute drop_connect tensor
        batch_size = net.get_shape().as_list()[0]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=tf.float32)
        binary_tensor = tf.floor(random_tensor)
        output = tf.div(net, keep_prob) * binary_tensor
        return output

    def MBConv2D(self, net, out_channels, exp_factor, stride, k_size, drop_connect_rate, in_channels=None):
        if in_channels is None:
            in_channels = net.get_shape().as_list()[0]
        residual = net
        net = Conv2D(in_channels * exp_factor, [1, 1], strides=[1, 1], padding='same')(net)
        net = BatchNormalization()(net)
        net = Activation('swish')(net)

        net = DepthwiseConv2D([k_size, k_size], strides=[stride, stride], padding='same')(net)
        net = BatchNormalization()(net)
        net = self._se_block(net)
        net = Activation('swish')(net)
        net = Conv2D(out_channels, [1, 1], strides=[1, 1], padding='same')(net)
        net = BatchNormalization()(net)
        if stride == 1 and in_channels == out_channels:
            if drop_connect_rate:
                net = Dropout(drop_connect_rate)(net)
            net = Add()([net, residual])
        return net

    def _mbconv_block(self, input_filters, net, output_filters,
                      kernel_size, stride,
                      expand_ratio, se_ratio,
                      id_skip, drop_connect_rate,
                      batch_norm_momentum=0.99,
                      batch_norm_epsilon=1e-3):

        has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        filters = input_filters * expand_ratio

        def block(inputs):

            if expand_ratio != 1:
                net = Conv2D(
                    filters,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    padding='same',
                    use_bias=False)(inputs)
                net = BatchNormalization()(net)
                net = Activation('swish')(net)
            else:
                net = inputs

            net = DepthwiseConv2D(
                [kernel_size, kernel_size],
                strides=[stride, stride],
                padding='same',
                use_bias=False)(net)
            net = BatchNormalization()(net)
            net = Activation('swish')(net)

            if has_se:
                net = self._se_block(net=net, exp_ratio=expand_ratio, se_ratio=se_ratio)(net)

            # output phase

            net = Conv2D(
                output_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same',
                use_bias=False)(net)
            net = BatchNormalization()(net)

            if id_skip:
                if all(s == 1 for s in stride) and (
                        input_filters == output_filters):

                    # only apply drop_connect if skip presents.
                    if drop_connect_rate:
                        x = self.drop_connect(drop_connect_rate)(x)

                    x = layers.Add()([x, inputs])

            return x

        return block

    def _se_block(self, net, exp_ratio, se_ratio=0.25):
        residual = net
        n_in = net.get_shape().as_list()[0]

        n_reduced_filter = max(1, int(n_in * se_ratio))
        in_channels = n_in * exp_ratio

        net = GlobalAveragePooling2D()(net)
        net = Lambda(lambda x: K.expand_dims(x, axis=1))(net)
        net = Lambda(lambda x: K.expand_dims(x, axis=1))(net)
        # reduce
        net = Conv2D(n_reduced_filter, [1, 1], strides=[1, 1], padding='same')(net)
        net = Activation('swish')(net)
        net = Conv2D(in_channels, [1, 1], strides=[1, 1], padding='same')(net)
        net = Activation('sigmoid')(net)
        return multiply([net, residual])

    @property
    def img_data(self):
        return self.data_size[1:]

    def build(self):
        data = Input(batch_shape=self.data_size)
        net = Conv2D(self._round_filters(32, self.width_coefficient), [3, 3], strides=[2, 2], padding='same')(data)
        net = BatchNormalization()(net)
        net = self._mbconv_block(in_channels=self._round_filters(32, self.width_coefficient),
                                 net=net, out_channels=self._round_filters(16, self.width_coefficient),
                                 num_layers=self._round_repeat(1, self.depth_coefficient),
                                 stride=1,
                                 exp_factor=1,
                                 k_size=3,
                                 drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(16, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(24, self.width_coefficient),
        #                          num_layers=self._round_repeat(2, self.depth_coefficient),
        #                          stride=2,
        #                          exp_factor=6,
        #                          k_size=3, drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(24, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(40, self.width_coefficient),
        #                          num_layers=self._round_repeat(2, self.depth_coefficient), stride=2,
        #                          exp_factor=6, k_size=5, drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(40, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(80, self.width_coefficient),
        #                          num_layers=self._round_repeat(3, self.depth_coefficient), stride=2,
        #                          exp_factor=6, k_size=3, drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(80, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(112, self.width_coefficient),
        #                          num_layers=self._round_repeat(3, self.depth_coefficient), stride=1,
        #                          exp_factor=6, k_size=5, drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(112, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(192, self.width_coefficient),
        #                          num_layers=self._round_repeat(4, self.depth_coefficient), stride=2,
        #                          exp_factor=6, k_size=5, drop_connect_rate=self.drop_connect_rate)
        # net = self._mbconv_block(in_channels=self._round_filters(192, self.width_coefficient),
        #                          net=net, out_channels=self._round_filters(320, self.width_coefficient),
        #                          num_layers=self._round_repeat(1, self.depth_coefficient), stride=1,
        #                          exp_factor=6, k_size=3, drop_connect_rate=self.drop_connect_rate)
        # net = Conv2D(self._round_filters(1280,self.width_coefficient),
        #              [1,1],strides=[1,1],padding='same')(net)
        # net = BatchNormalization()(net)
        # net = GlobalAveragePooling2D()(net)
        # net = Dropout(self.dropout_rate)(net)
        # net = Dense(self.n_classes)(net)
        model = Model(inputs=data, outputs=net)
        model.summary()
        return net

    # def get_efficient_net(width_coefficient, depth_coefficient, resolution, dropout_rate):
    #     net = EfficientNet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)

    @property
    def efficient_net_b0(width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):  # 1.0, 1.0, 224, 0.2
        return efficientnet(width_coefficient=width_coefficient,
                            depth_coefficient=depth_coefficient,
                            dropout_rate=dropout_rate)

    # @property
    # def efficient_net_b1(1.0, 1.1, 240, 0.2):
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)
    # @property
    # def efficient_net_b2(1.1, 1.2, 260, 0.3):
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)
    #
    # @property
    # def efficient_net_b3(1.2, 1.4, 300, 0.3):
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)
    #
    # @property
    # def efficient_net_b4(1.4, 1.8, 380, 0.4):
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)
    #
    # @property
    # def efficient_net_b5(1.6, 2.2, 456, 0.4):
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)
    #
    # @property
    # def efficient_net_b6(width_coefficient=1.8, depth_coefficient=2.6, 528, dropout_rate=0.5):  # 1.8, 2.6, 528, 0.5
    #     return efficientnet(width_coefficient=width_coefficient,
    #                        depth_coefficient=depth_coefficient,
    #                        dropout_rate=dropout_rate)


def efficient_net_b7(width_coefficient=2.0, depth_coefficient=3.1, dropout_rate=0.5):  # 2.0, 3.1, 600, 0.5
    return efficientnet(width_coefficient=width_coefficient,
                        depth_coefficient=depth_coefficient,
                        dropout_rate=dropout_rate).build()


efficient_net_b7()
