# -*- coding: utf-8 -*-

import tensorflow as tf
from util.LayerUtil import LayerUtil
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# 导入MNIST数据集
mnist = input_data.read_data_sets("D:\\tensorflow\\cha5\\mnist_data", one_hot=True)

# 定义超参数
learning_rate = 0.01
training_epochs = 10
batch_size = 256
display_step = 1

# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])


def encoder(X):
    x, y = X.get_shape().as_list()
    layer1 = LayerUtil.add_layer(X, y, n_hidden_1, 'encoder_layer1', active_function=tf.nn.sigmoid)
    layer2 = LayerUtil.add_layer(layer1, n_hidden_1, n_hidden_2, 'encoder_layer2', active_function=tf.nn.sigmoid)
    layer3 = LayerUtil.add_layer(layer2, n_hidden_2, n_hidden_3, 'encoder_layer3', active_function=tf.nn.sigmoid)
    layer4 = LayerUtil.add_layer(layer3, n_hidden_3, n_hidden_4, 'encoder_layer4', active_function=None)
    return layer4


def decoder(X):
    x, y = X.get_shape()
    layer1 = LayerUtil.add_layer(X, n_hidden_4, n_hidden_3, 'decoder_layer1', active_function=tf.nn.sigmoid)
    layer2 = LayerUtil.add_layer(layer1, n_hidden_3, n_hidden_2, 'decoder_layer2', active_function=tf.nn.sigmoid)
    layer3 = LayerUtil.add_layer(layer2, n_hidden_2, n_hidden_1, 'decoder_layer3', active_function=tf.nn.sigmoid)
    layer4 = LayerUtil.add_layer(layer3, n_hidden_1, 784, 'decoder_layer4', active_function=tf.nn.sigmoid)
    return layer4


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            if i % display_step == 0:
                print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
        row, col=np.shape(mnist.test.labels)
        color_list=[]
        for x in range(row):
            for y in range(col):
                if mnist.test.labels[x,y] == 1 :
                    color_list.append(y)

        plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=color_list)
        plt.colorbar()
        plt.show()
    pass
