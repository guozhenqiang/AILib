# -*- coding: utf-8 -*-

import tensorflow as tf

class LayerUtil():

    def __init__(self):
        pass

    # 定义一个方法可以用来添加网络层
    @staticmethod
    def add_layer(input, input_size, output_size, n_layer, active_function=None):
        #为每个层指定一个名称
        layer_name = 'layer%s' % n_layer
        #with tf.name_scope()这样是为了在tensorboard可视化，其中每个变量需要指定变量名
        with tf.name_scope('layer'):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.random_normal([input_size, output_size]), name='W')
                #记录变量的变化情况到tensorboard中
                tf.summary.histogram(layer_name + '/weights', Weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, output_size]) + 0.1, name='b')
                tf.summary.histogram(layer_name + '/biases', biases)
            with tf.name_scope('y'):
                y = tf.add(tf.matmul(input, Weights), biases)
            if active_function is None:
                outputs = y
            else:
                outputs = active_function(y)
            tf.summary.histogram(layer_name + '/outputs', outputs)
            return outputs

