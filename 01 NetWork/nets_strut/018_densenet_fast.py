from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Activation, \
    AveragePooling2D, GlobalAveragePooling2D, concatenate
from keras.layers import Input
from keras.layers import BatchNormalization

'''
Based on the implementation here : https://github.com/Lasagne/Recipes/blob/master/papers/densenet/densenet_fast.py
'''


class densenet_fast():
    def __init__(self, nb_classes, batch_size, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16,
                 dropout_rate=None,
                 weight_decay=1E-4):
        self.nb_classes = nb_classes
        self.data_size = [batch_size, 224, 224, 3]
        self.depth = depth
        self.nb_dense_block = nb_dense_block
        self.growth_rate = growth_rate
        self.nb_filter = nb_filter
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.concat_axis = -1
        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
        # layers in each dense block
        self.nb_layers = int((depth - 4) / 3)

    def conv_block(self, net, nb_filter, dropout_rate=None):
        net = Activation('relu')(net)
        net = Conv2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False)(net)
        if dropout_rate:
            net = Dropout(dropout_rate)(net)
        return net

    def transition_block(self, net, nb_filter, dropout_rate=None):
        concat_axis = -1
        net = Conv2D(nb_filter, 1, 1, init="he_uniform", border_mode="same", bias=False)(net)
        if dropout_rate:
            net = Dropout(dropout_rate)(net)
        net = AveragePooling2D((2, 2), strides=(2, 2))(net)
        net = BatchNormalization(mode=0, axis=concat_axis)(net)
        return net

    def dense_block(self, net, nb_layers, nb_filter, growth_rate, dropout_rate=None):
        concat_axis = -1
        feature_list = [net]
        for i in range(nb_layers):
            net = self.conv_block(net, growth_rate, dropout_rate)
            feature_list.append(net)
            net = concatenate(feature_list, axis=concat_axis)
            nb_filter += growth_rate
        return net, nb_filter

    def build(self):
        # Initial convolution
        data = Input(batch_shape=self.data_size)
        net = Conv2D(self.nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                   )(data)
        net = BatchNormalization(mode=0, axis=self.concat_axis)(net)

        # Add dense blocks
        for block_idx in range(self.nb_dense_block - 1):
            net, nb_filter = self.dense_block(net, self.nb_layers, self.nb_filter, self.growth_rate,
                                            dropout_rate=self.dropout_rate)
            # add transition_block
            net = self.transition_block(net, nb_filter, dropout_rate=self.dropout_rate)

        # The last dense_block does not have a transition_block
        net, nb_filter = self.dense_block(net, self.nb_layers, self.nb_filter, self.growth_rate,
                                        dropout_rate=self.dropout_rate)

        net = Activation('relu')(net)
        net = GlobalAveragePooling2D()(net)
        net = Dense(self.nb_classes, activation='softmax')(net)

        model = Model(input=data, output=net, name="DenseNet-%d-%d created." % (self.depth, self.growth_rate))
        model.summary()

        return net


densenet_fast(1000, 32).build()
