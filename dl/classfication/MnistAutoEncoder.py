# -*- coding: utf-8 -*-

import tensorflow as tf
from util.LayerUtil import LayerUtil
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

#导入MNIST数据集
mnist = input_data.read_data_sets("D:\\tensorflow\\cha5\\mnist_data", one_hot=True)

#定义模型的超参数
learning_rate = 0.01
training_epochs = 5 # 五组训练
batch_size = 256
display_step = 1
examples_to_show = 10

n_input = 784

n_hidden_1 = 256
n_hidden_2 = 128


X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])


def encoder(X):
    layer1 = LayerUtil.add_layer(X, 784, 256, 'encoder_layer1', active_function=tf.nn.sigmoid)
    layer2 = LayerUtil.add_layer(layer1, 256, 128, 'encoder_layer2', active_function=tf.nn.sigmoid)
    return layer2


def decoder(X):
    layer1 = LayerUtil.add_layer(X, 128, 256, 'decoder_layer1', active_function=tf.nn.sigmoid)
    layer2 = LayerUtil.add_layer(layer1, 256, 784, 'decoder_layer2', active_function=tf.nn.sigmoid)
    return layer2


if __name__ == '__main__':

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op
    y_true = X

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        batch_sum =  int(mnist.train.num_examples/batch_size)

        for i in range(batch_sum):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            if i % display_step == 0 :
                print("Epoch:", '%04d' % (i + 1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # # Applying encode and decode over test set
        encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        plt.show()



    pass

