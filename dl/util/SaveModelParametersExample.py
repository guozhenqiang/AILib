# -*- coding: utf-8 -*-

"""
主要介绍tensorflow如何保存训练好的模型中的参数，以及获取保存的参数到模型中用于预测
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

def save_parameter(path):
    # 保存参数变量到文件中
    W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

    init = tf.global_variables_initializer()
    # tf.train.Saver()用于实现保存
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        save_path = saver.save(sess, path)
        print("Save to path: ", save_path)


def read_parameter(path):
    # 读取保存到文件中的参数变量
    W_ = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b_ = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    # 读取本地参数变量的时候，不需要初始化了

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path)
        print("weights:", sess.run(W_))
        print("biases:", sess.run(b_))



if __name__ == '__main__':
    # path中的路径必须存在
    path='mynet\\save_net.ckpt'
    # save_parameter(path)
    read_parameter(path)
