# -*- coding: utf-8 -*-

import tensorflow as tf

class LayerUtil():

    def __init__(self):
        pass

    # 定义一个方法可以用来添加网络层
    @staticmethod
    def add_layer(input, input_size, output_size, active_function=None):
        Weights = tf.Variable(tf.random_normal([input_size, output_size]))
        biases = tf.Variable(tf.zeros([1, output_size]) + 0.1)
        y = tf.matmul(input, Weights) + biases
        if active_function is None:
            result = y
        else:
            result = active_function(y)
        return result
