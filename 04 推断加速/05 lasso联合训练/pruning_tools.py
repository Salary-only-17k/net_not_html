import numpy as np
import tensorflow as tf

def _L1_fs(tensor):
    return tf.reduce_sum(tensor)


def _L1_2_fs(tensor):
    return tf.sqrt(tf.reduce_sum(tensor))


def _L2_fs(tensor):
    return tf.reduce_sum(tf.pow(tensor,2))


def lasso_out(y, pre_y, ratio=0.2, func=_L1_fs):
    lasso_loss = tf.losses.mean_squared_error(y, pre_y)+ratio * func(pre_y)
    tf.train.GradientDescentOptimizer(lasso_loss)

