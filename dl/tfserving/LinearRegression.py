# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np



# 定义超参
learning_rate = 0.01
batch_size = 100
n_steps = 1000
n_samples = 1000

x_data = np.arange(0, 100, 0.1)
y_data = x_data * 2 + 5

x_ = np.reshape(x_data, (n_samples, 1))
y_ = np.reshape(y_data, (n_samples, 1))

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

Weights = tf.Variable(tf.random_normal([1,1]), name='w')
biases = tf.Variable(tf.zeros([1,]) + 0.1, name='b')

prediction = tf.matmul(x, Weights)+biases
cost = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

def train_model(path):
    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_steps):
            indices = np.random.choice(n_samples, batch_size)
            x_batch = x_[indices]
            y_batch = y_[indices]
            _, cost_val = sess.run([train, cost], feed_dict={x: x_batch, y: y_batch})
            if i % 50 == 0:
                print(i+1, sess.run(cost, feed_dict={x: x_, y: y_}))
        saver.save(sess, path)
        print(Weights.eval())
        print(biases.eval())


def use_model(path, input):
    input = np.reshape(input, (1,1))
    with tf.Session() as sess:
        saver.restore(sess, path)
        # sess.run(Weights)
        # sess.run(biases)
        return sess.run(prediction, feed_dict={x: input})


if __name__ == '__main__':
    path='d:\\tensorflow\\model\\lr.ckpt'
    train_model(path)
    input = np.arange(1)
    print(use_model(path, input))
    pass

