import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.layers as ly
import math as m

from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D
from keras.layers import Dropout, BatchNormalization, Activation
from keras import regularizers
from keras.models import Model
import keras.activations as activation
'''
原味Lenet，没有bn，dropout之类的处理
2020229
chenglong
'''
def mergeConvBn_tf(net,weight_init,bias_init):
    beta = 1E-5
    axis = list(range(len(net.get_shape().as_list()) - 1))
    mean, variance = tf.nn.moments(net, axis)
    alpha = 1/m.sqrt(variance+beta)
    weights = tf.multiply(weight_init,alpha)
    biases = alpha*bias_init +mean*alpha

    return weights,biases



def mergeConvBn_k(net,weight_init,bias_init):
    beta = 1E-5
    axis = list(range(len(net.get_shape().as_list()) - 1))
    mean, variance = K.mean(net, axis),K.var(net, axis)
    alpha = 1/m.sqrt(variance+beta)
    weights = K.dot(weight_init,alpha)
    biases = alpha*bias_init + mean/m.sqrt(variance)
    return weights,biases


# def mergeConvBn():
#     def inner(net):
#         beta = 1E-5
#         axis = list(range(len(net.get_shape().as_list())))
#         mean, var = K.mean(net, axis), K.var(net, axis)
#         alpha = 1 / K.sqrt(var + beta)
#         return alpha, mean, var
#
#     return inner
#
#
# def weight_init(shape, dtype=None):
#     nonlocal alpha, mean
#     return K.random_normal(shape, dtype=dtype) * alpha
#
#
# def bias_init(shape=1, dtype=None):
#     nonlocal alpha, mean
#     return alpha * K.ones(shape, dtype=dtype) + mean / alpha








