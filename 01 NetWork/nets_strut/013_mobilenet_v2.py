import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from keras.models import Model
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, DepthwiseConv2D,ReLU
from keras.layers import Dense, Input, BatchNormalization, Activation, Add,Flatten, DepthwiseConv2D
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
import keras.backend as K
from keras.utils import get_custom_objects
import tensorflow as tf
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.engine import InputSpec

# #
# class DepthwiseConv2D(Conv2D):
#     """Depthwise separable 2D convolution.
#
#     Depthwise Separable convolutions consists in performing
#     just the first step in a depthwise spatial convolution
#     (which acts on each input channel separately).
#     The `depth_multiplier` argument controls how many
#     output channels are generated per input channel in the depthwise step.
#
#     # Arguments
#         kernel_size: An integer or tuple/list of 2 integers, specifying the
#             width and height of the 2D convolution window.
#             Can be a single integer to specify the same value for
#             all spatial dimensions.
#         strides: An integer or tuple/list of 2 integers,
#             specifying the strides of the convolution along the width and height.
#             Can be a single integer to specify the same value for
#             all spatial dimensions.
#             Specifying any stride value != 1 is incompatible with specifying
#             any `dilation_rate` value != 1.
#         padding: one of `'valid'` or `'same'` (case-insensitive).
#         depth_multiplier: The number of depthwise convolution output channels
#             for each input channel.
#             The total number of depthwise convolution output
#             channels will be equal to `filters_in * depth_multiplier`.
#         data_format: A string,
#             one of `channels_last` (default) or `channels_first`.
#             The ordering of the dimensions in the inputs.
#             `channels_last` corresponds to inputs with shape
#             `(batch, height, width, channels)` while `channels_first`
#             corresponds to inputs with shape
#             `(batch, channels, height, width)`.
#             It defaults to the `image_data_format` value found in your
#             Keras config file at `~/.keras/keras.json`.
#             If you never set it, then it will be 'channels_last'.
#         activation: Activation function to use
#             (see [activations](../activations.md)).
#             If you don't specify anything, no activation is applied
#             (ie. 'linear' activation: `a(x) = x`).
#         use_bias: Boolean, whether the layer uses a bias vector.
#         depthwise_initializer: Initializer for the depthwise kernel matrix
#             (see [initializers](../initializers.md)).
#         bias_initializer: Initializer for the bias vector
#             (see [initializers](../initializers.md)).
#         depthwise_regularizer: Regularizer function applied to
#             the depthwise kernel matrix
#             (see [regularizer](../regularizers.md)).
#         bias_regularizer: Regularizer function applied to the bias vector
#             (see [regularizer](../regularizers.md)).
#         activity_regularizer: Regularizer function applied to
#             the output of the layer (its 'activation').
#             (see [regularizer](../regularizers.md)).
#         depthwise_constraint: Constraint function applied to
#             the depthwise kernel matrix
#             (see [constraints](../constraints.md)).
#         bias_constraint: Constraint function applied to the bias vector
#             (see [constraints](../constraints.md)).
#
#     # Input shape
#         4D tensor with shape:
#         `[batch, channels, rows, cols]` if data_format='channels_first'
#         or 4D tensor with shape:
#         `[batch, rows, cols, channels]` if data_format='channels_last'.
#
#     # Output shape
#         4D tensor with shape:
#         `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
#         or 4D tensor with shape:
#         `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
#         `rows` and `cols` values might have changed due to padding.
#     """
#
#     def __init__(self,
#                  kernel_size,
#                  strides=(1, 1),
#                  padding='valid',
#                  depth_multiplier=1,
#                  data_format=None,
#                  activation=None,
#                  use_bias=True,
#                  depthwise_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  depthwise_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  depthwise_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         super(DepthwiseConv2D, self).__init__(
#             filters=None,
#             kernel_size=kernel_size,
#             strides=strides,
#             padding=padding,
#             data_format=data_format,
#             activation=activation,
#             use_bias=use_bias,
#             bias_regularizer=bias_regularizer,
#             activity_regularizer=activity_regularizer,
#             bias_constraint=bias_constraint,
#             **kwargs)
#         self.depth_multiplier = depth_multiplier
#         self.depthwise_initializer = initializers.get(depthwise_initializer)
#         self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
#         self.depthwise_constraint = constraints.get(depthwise_constraint)
#         self.bias_initializer = initializers.get(bias_initializer)
#
#     def build(self, input_shape):
#         if len(input_shape) < 4:
#             raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
#                              'Received input shape:', str(input_shape))
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = 3
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs to '
#                              '`DepthwiseConv2D` '
#                              'should be defined. Found `None`.')
#         input_dim = int(input_shape[channel_axis])
#         depthwise_kernel_shape = (self.kernel_size[0],
#                                   self.kernel_size[1],
#                                   input_dim,
#                                   self.depth_multiplier)
#
#         self.depthwise_kernel = self.add_weight(
#             shape=depthwise_kernel_shape,
#             initializer=self.depthwise_initializer,
#             name='depthwise_kernel',
#             regularizer=self.depthwise_regularizer,
#             constraint=self.depthwise_constraint)
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # Set input spec.
#         self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
#         self.built = True
#
#     def call(self, inputs, training=None):
#         outputs = K.depthwise_conv2d(
#             inputs,
#             self.depthwise_kernel,
#             strides=self.strides,
#             padding=self.padding,
#             dilation_rate=self.dilation_rate,
#             data_format=self.data_format)
#
#         if self.bias:
#             outputs = K.bias_add(
#                 outputs,
#                 self.bias,
#                 data_format=self.data_format)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#
#         return outputs

class mobilenet_v2():
    def __init__(self, batch_size=32, n_classes=1000, flags=1, alpha=1.0):
        self.data_size = [batch_size, 224, 224, 3]
        self.n_classes = n_classes
        self.flags = flags
        self.alpha = alpha

    def build(self):
        data = Input(batch_shape=self.data_size)
        first_block_n_out = self._make_divisible(32 * self.alpha, 8)
        net = Conv2D(first_block_n_out, [3, 3], strides=[2, 2], padding='same',
                     use_bias=False, activation=None, name='conv1')(data)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)
        net = ReLU(max_value=6.0)(net)

        net = self._first_inverted_res_block(net, 1, self.alpha, 16, 0)
        net = self._inverted_res_block(24, net, 6, 2, self.alpha, 1)
        net = self._inverted_res_block(24, net, 6, 1, self.alpha, 2)

        net = self._inverted_res_block(32, net, 6, 2, self.alpha, 3)
        net = self._inverted_res_block(32, net, 6, 1, self.alpha, 4)
        net = self._inverted_res_block(32, net, 6, 1, self.alpha, 5)

        net = self._inverted_res_block(64, net, 6, 2, self.alpha, 6)
        net = self._inverted_res_block(64, net, 6, 1, self.alpha, 7)
        net = self._inverted_res_block(64, net, 6, 1, self.alpha, 8)
        net = self._inverted_res_block(64, net, 6, 1, self.alpha, 9)

        net = self._inverted_res_block(96, net, 6, 2, self.alpha, 10)
        net = self._inverted_res_block(96, net, 6, 1, self.alpha, 11)
        net = self._inverted_res_block(96, net, 6, 1, self.alpha, 12)

        net = self._inverted_res_block(160, net, 6, 2, self.alpha, 13)
        net = self._inverted_res_block(160, net, 6, 2, self.alpha, 14)
        net = self._inverted_res_block(160, net, 6, 2, self.alpha, 15)

        net = self._inverted_res_block(320, net, 6, 2, self.alpha, 16)
        if self.alpha >1.0:
            last_block_filters = self._make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280
        net = Conv2D(last_block_filters,[1,1],strides=[1,1],use_bias=False,name='conv_last')(net)
        net = ReLU(max_value=6.0)(net)


        net = GlobalAveragePooling2D()(net)
        net = Dense(self.n_classes, activation='softmax',
                  use_bias=True, name='Logits')(net)


        # Create model.
        model = Model(inputs=data, outputs=net,name=f'mobilenet_v2_{self.alpha}_')
        model.summary()
        return net


    def preprocess_input(self, x):
        x /= 128.
        x -= 1.
        return x.astype(tf.float32)

    def unprocess_input(self, x):
        """Unprocesses a numpy array encoding a batch of images.
        This function undoes the preprocessing which converts
        the RGB values from [0, 255] to [-1, 1].

        # Arguments
            x: a 4D numpy array consists of RGB values within [0, 255].

        # Returns
            Preprocessed array.
        """
        x += 1.
        x *= 128.
        return x.astype(tf.uint8)

    def _make_divisible(self, value, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(value + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        # 确保向下舍入的幅度不超过10%。
        if new_v < 0.9 * value:
            new_v += divisor
        return new_v

    def _inverted_res_block(self, n_out, net, expansion, stride, alpha, block_id):
        snet = net
        n_in = net.get_shape().as_list()[-1]
        prefix = 'features.' + str(block_id) + '.conv.'
        pointwise_conv_filters = int(n_out * alpha)
        pointwise_filters = self._make_divisible(pointwise_conv_filters, 8)
        # Expand

        net = Conv2D(expansion * n_in, kernel_size=1, padding='same',
                     use_bias=False, activation=None,
                     name='mobl%d_conv_expand' % block_id)(net)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)
        net = ReLU(max_value=6.0)(net)

        # Depthwise
        net = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                              use_bias=False, padding='same',
                              name='mobl%d_conv_depthwise' % block_id)(net)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)

        net = ReLU(max_value=6.0)(net)

        # Project
        net = Conv2D(pointwise_filters,
                     kernel_size=1, padding='same', use_bias=False, activation=None,
                     name='mobl%d_conv_project' % block_id)(net)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)

        if n_in == pointwise_filters and stride == 1:
            return Add(name='res_connect_' + str(block_id))([snet, net])

        return net

    def _first_inverted_res_block(self, net, stride, alpha, n_out, block_id):
        # alpha放大系数
        # 他与别的区别是 不是conv开头
        snet = net
        n_in = net.get_shape().as_list()[-1]
        pointwise_conv_filters = int(n_out * alpha)
        pointwise_n_out = self._make_divisible(pointwise_conv_filters, 8)

        net = DepthwiseConv2D([3, 3], strides=[stride, stride], activation=None,
                              use_bias=False, padding='same',
                              name='mobl%d_conv_depthwise' % block_id)(net)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)
        net = ReLU(max_value=6.0)(net)

        net = Conv2D(pointwise_n_out, [1, 1], strides=[1, 1],
                     padding='same', activation=None,use_bias=False,
                     name='mobl%d_conv_project' % block_id)(net)
        net = BatchNormalization(epsilon=1e-3, momentum=0.999)(net)

        if n_in == pointwise_n_out and stride == 1:
            return Add(name='res_connect_' + str(block_id))([snet, net])
        return net


mobilenet_v2().build()
