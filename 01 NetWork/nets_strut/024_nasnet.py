"""Implementation of NASNet-A"""
from keras import Input, Model
from keras import backend as K
from keras.engine import get_source_inputs
from keras.layers import Activation, SeparableConv2D, BatchNormalization, Dropout, ZeroPadding2D, \
    GlobalMaxPooling2D, Cropping2D, Convolution2D, GlobalAveragePooling2D, Dense, AveragePooling2D, \
    MaxPooling2D, Add, Concatenate


class NASNetA():
    def __init__(self,
                 include_top=True,
                 batch_size=None,
                 pooling=None,
                 add_aux_output=False,
                 stem=None,
                 stem_filters=96,
                 num_cell_repeats=18,
                 penultimate_filters=768,
                 num_classes=10,
                 num_reduction_cells=2,
                 dropout_rate=0.5):
        self.add_aux_output = add_aux_output
        self.include_top = include_top
        self.input_tensor = [batch_size, 224, 224, 3]
        self.dropout_rate = dropout_rate
        self.aux_outputs = None
        self.stem_filters = stem_filters
        self.num_cell_repeats = num_cell_repeats
        self.num_classes = num_classes
        self.num_reduction_cells = num_reduction_cells
        if pooling is None:
            self.pooling = 'avg'

        if stem is None:
            self.stem = self._ImagenetStem
        self.penultimate_filters = penultimate_filters
        self.filters = int(penultimate_filters / ((2 ** num_reduction_cells) * 6))

    def _ImagenetStem(self, net, stem_filters, filters):
        with K.name_scope('imagenet-stem'):
            net = Convolution2D(stem_filters, 3, strides=2,
                                kernel_initializer='he_normal', padding='valid', use_bias=False,
                                name='conv0')(net)
            net = BatchNormalization(name='conv0_bn')(net)

            prev = self._ReductionCell(filters=filters // 4, prefix='cell_stem_0', prev=None, cur=net)
            cur = self._ReductionCell(filters=filters // 2, prefix='cell_stem_1', prev=net, cur=prev)

        return prev, cur

    def _Separable(self, net, filters, kernel_size, prefix, strides=1):
        with K.name_scope('separable_{0}x{0}_strides_{1}'.format(kernel_size, strides)):
            for repeat in range(1, 3):
                strides = strides if repeat == 1 else 1

                net = Activation('relu')(net)

                name = '{0}/separable_{1}x{1}_{2}'.format(prefix, kernel_size, repeat)
                net = SeparableConv2D(filters,
                                      kernel_size=kernel_size,
                                      kernel_initializer='he_normal',
                                      strides=strides,
                                      padding='same',
                                      use_bias=False,
                                      name=name)(net)

                name = '{0}/bn_sep_{1}x{1}_{2}'.format(prefix, kernel_size, repeat)
                net = BatchNormalization(name=name, axis=-1)(net)

        return net

    def _SqueezeChannels(self, x, filters, prefix, conv_suffix='1x1', bn_suffix='beginning_bn'):
        """Use 1x1 convolutions to squeeze the input channels to match the cells filter count"""

        conv_name = '{}/{}'.format(prefix, conv_suffix)
        bn_name = '{}/{}'.format(prefix, bn_suffix)
        with K.name_scope('filter_squeeze'):
            x = Activation('relu')(x)
            x = Convolution2D(filters, 1, kernel_initializer='he_normal', use_bias=False,
                              name=conv_name)(x)
            x = BatchNormalization(name=bn_name, axis=-1)(x)

            return x

    def _Fit(self, net, filters, target_layer, prefix):
        """Make the cell outputs compatible"""
        if net is None:
            return target_layer

        elif int(net.shape[2]) != int(target_layer.shape[2]):

            with K.name_scope('reduce_shape'):
                net = Activation('relu')(net)

                p1 = AveragePooling2D(pool_size=1, strides=(2, 2), padding='valid')(net)
                p1 = Convolution2D(filters // 2,
                                   kernel_size=1,
                                   kernel_initializer='he_normal',
                                   padding='same',
                                   use_bias=False,
                                   name='{}/path1_conv'.format(prefix))(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(net)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D(pool_size=1, strides=2, padding='valid')(p2)

                p2 = Convolution2D(filters // 2,
                                   kernel_size=1,
                                   kernel_initializer='he_normal',
                                   padding='same',
                                   use_bias=False,
                                   name='{}/path2_conv'.format(prefix))(p2)

                net = Concatenate(axis=-1)([p1, p2])
                net = BatchNormalization(name='{}/final_path_bn'.format(prefix), axis=-1)(net)

                return net
        else:
            return self._SqueezeChannels(filters=filters, prefix=prefix, conv_suffix='prev_1x1', bn_suffix='prev_bn',
                                         x=x)

    def _NormalCell(self, filters, prefix, prev, cur):

        with K.name_scope('normal'):
            cur = self._SqueezeChannels(filters=filters, prefix=prefix, x=cur)
            prev = self._Fit(filters=filters, target_layer=cur, prefix=prefix, net=prev)
            output = [prev]

            with K.name_scope('comb_iter_0'):
                prefix = '{}/comb_iter_0'.format(prefix)
                output.append(
                    Add()([self._Separable(filters=filters, kernel_size=5, prefix='{}/left'.format(prefix), net=cur),
                           self._Separable(filters=filters, kernel_size=3, prefix='{}/right'.format(prefix),
                                           net=prev)]))

            with K.name_scope('comb_iter_1'):
                prefix = '{}/comb_iter_1'.format(prefix)
                output.append(
                    Add()([self._Separable(filters=filters, kernel_size=5, prefix='{}/left'.format(prefix), net=prev),
                           self._Separable(filters=filters, kernel_size=3, prefix='{}/right'.format(prefix),
                                           net=prev)]))

            with K.name_scope('comb_iter_2'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(cur),
                                     prev]))

            with K.name_scope('comb_iter_3'):
                output.append(Add()([AveragePooling2D(pool_size=3, strides=1, padding='same')(prev),
                                     AveragePooling2D(pool_size=3, strides=1, padding='same')(prev)]))

            with K.name_scope('comb_iter_4'):
                prefix = '{}/comb_iter_4'.format(prefix)
                output.append(
                    Add()([self._Separable(filters=filters, kernel_size=3, prefix='{}/left'.format(prefix), net=cur),
                           cur]))

            return Concatenate(axis=-1)(output)

    def _ReductionCell(self, filters, prefix, prev, cur):
        with K.name_scope('reduce'):
            prev = self._Fit(filters=filters, target_layer=cur, prefix=prefix, net=prev)
            cur = self._SqueezeChannels(filters=filters, prefix=prefix, x=cur)

            # Full in
            with K.name_scope('comb_iter_0'):
                prefix = '{}/comb_iter_0'.format(prefix)
                add_0 = Add()([self._Separable(filters=filters, kernel_size=5, strides=2,
                                               prefix='{}/left'.format(prefix), net=cur),
                               self._Separable(filters=filters, kernel_size=7, strides=2,
                                               prefix='{}/right'.format(prefix), net=prev)])

            with K.name_scope('comb_iter_1'):
                prefix = '{}/comb_iter_1'.format(prefix)
                add_1 = Add()([MaxPooling2D(3, strides=2, padding='same')(cur),
                               self._Separable(filters=filters, kernel_size=7, strides=2,
                                               prefix='{}/right'.format(prefix), net=prev)])

            with K.name_scope('comb_iter_2'):
                prefix = '{}/comb_iter_2'.format(prefix)
                add_2 = Add()([AveragePooling2D(3, strides=2, padding='same')(cur),
                               self._Separable(filters=filters, kernel_size=5, strides=2,
                                               prefix='{}/right'.format(prefix), net=prev)])

            # Reduced after stride
            with K.name_scope('comb_iter_3'):
                add_3 = Add()([AveragePooling2D(3, strides=1, padding='same')(add_0), add_1])

            with K.name_scope('comb_iter_4'):
                prefix = '{}/comb_iter_4'.format(prefix)
                add_4 = Add()([self._Separable(filters=filters, kernel_size=3, strides=1,
                                               prefix='{}/left'.format(prefix), net=add_0),
                               MaxPooling2D(3, strides=2, padding='same')(cur)])

            return Concatenate(axis=-1)([add_1, add_2, add_3, add_4])

    def _AuxiliaryTop(self, net, classes, prefix):
        prefix = '{}/aux_logits'.format(prefix)
        with K.name_scope('auxiliary_output'):
            net = Activation('relu')(net)
            net = AveragePooling2D(5, strides=3, padding='valid')(net)
            net = Convolution2D(128, kernel_size=1, padding='same',
                                kernel_initializer='he_normal', use_bias=False,
                                name='{}/proj'.format(prefix))(net)
            net = BatchNormalization(name='{}/aux_bn0'.format(prefix), axis=-1)(net)

            net = Activation('relu')(net)
            net = Convolution2D(768, kernel_size=int(net.shape[2]), padding='valid',
                                kernel_initializer='he_normal', use_bias=False,
                                name='{}/Conv'.format(prefix))(net)
            net = BatchNormalization(name='{}/aux_bn1'.format(prefix), axis=-1)(net)

            net = Activation('relu')(net)
            net = GlobalAveragePooling2D()(net)

            net = Dense(classes, activation='softmax', name='{}/FC'.format(prefix))(net)

        return net

    def build(self):
        inputs = Input(batch_shape=self.input_tensor)
        prev, cur = self.stem(filters=self.filters, stem_filters=self.stem_filters, net=inputs)

        for repeat in range(self.num_reduction_cells + 1):
            if repeat == self.num_reduction_cells and self.add_aux_output:
                prefix = 'aux_{}'.format(repeat * self.num_cell_repeats - 1)
                aux_outputs = self._AuxiliaryTop(classes=self.num_classes, prefix=prefix, net=cur)

            if repeat > 0:
                self.filters *= 2
                prev, cur = cur, prev
                cur = self._ReductionCell(filters=self.filters, prefix='reduction_cell_{}'.format(repeat - 1), cur=prev,
                                          prev=cur)

            for cell_index in range(self.num_cell_repeats):
                prev, cur = cur, prev
                cur = self._NormalCell(filters=self.filters,
                                       prefix='cell_{}'.format(cell_index + repeat * self.num_cell_repeats), cur=prev,
                                       prev=cur)

        with K.name_scope('final_layer'):
            x = Activation('relu', name='last_relu')(cur)

            if self.include_top:
                x = GlobalAveragePooling2D(name='avg_pool')(x)
                x = Dropout(rate=self.dropout_rate)(x)
                outputs = Dense(self.num_classes, activation='softmax', name='final_layer/FC')(x)

                model_suffix = 'with_top'
            else:
                if self.pooling == 'avg':
                    outputs = GlobalAveragePooling2D(name='avg_pool')(x)
                elif self.pooling == 'max':
                    outputs = GlobalMaxPooling2D(name='max_pool')(x)
                else:
                    outputs = None
                    raise Exception(
                        'Supported options for pooling: `avg` or `max` given pooling: {}'.format(self.pooling))

                model_suffix = 'no_top'
        model_name = 'NASNet-A_{}@{}_{}_{}'.format(self.num_cell_repeats, self.penultimate_filters, self.num_classes,
                                                   model_suffix)
        if self.add_aux_output:
            model = Model(inputs, [outputs, aux_outputs], name='{}_with_auxiliary_output'.format(model_name))
            model.summary()
            return model
        else:
            model = Model(inputs, outputs, name=model_name)
            model.summary()
            return model


def mobile(include_top=True,
           input_tensor=None,
           aux_output=False,
           num_classes=1000,
           ):
    """Table 3: NASNet-A (4 @ 1056), 5.3M parameters"""
    batch_size = 32

    model = NASNetA(include_top=include_top,
                    batch_size=batch_size,
                    num_cell_repeats=4,
                    add_aux_output=aux_output,
                    stem_filters=32,
                    penultimate_filters=1056,
                    num_classes=num_classes).build()

    return model


if __name__ == '__main__':
    model = mobile()
