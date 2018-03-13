# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
#定义一些常量
N_SAMPLES = 20
N_HIDDEN = 300
LR = 0.01
#构造训练集
x_train = np.linspace(-1, 1, N_SAMPLES)[:, np.newaxis]
y_train = x_train+0.3*np.random.rand(N_SAMPLES)[:, np.newaxis]
#构造测试集
x_test=x_train.copy()
y_test=x_test+0.3*np.random.rand(N_SAMPLES)[:, np.newaxis]
#显示原始数据
plt.scatter(x_train, y_train, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 定义placeholders
tf_x = tf.placeholder(tf.float32, [None, 1])
tf_y = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

# overfitting net
o1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
o2 = tf.layers.dense(o1, N_HIDDEN, tf.nn.relu)
o_out = tf.layers.dense(o2, 1)
o_loss = tf.losses.mean_squared_error(tf_y, o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# dropout net
d1 = tf.layers.dense(tf_x, N_HIDDEN, tf.nn.relu)
d1 = tf.layers.dropout(d1, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d2 = tf.layers.dense(d1, N_HIDDEN, tf.nn.relu)
d2 = tf.layers.dropout(d2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs
d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(tf_y, d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        plt.ion()  # something about plotting
        for t in range(500):
            sess.run([o_train, d_train], {tf_x: x_train, tf_y: y_train, tf_is_training: True})

            if t % 10 == 0:
                plt.cla()
                o_loss_, d_loss_, o_out_, d_out_ = sess.run(
                    [o_loss, d_loss, o_out, d_out], {tf_x: x_test, tf_y: y_test, tf_is_training: False}
                )
                plt.scatter(x_train, y_train, c='magenta', s=50, alpha=0.3, label='train');
                plt.scatter(x_test, y_test, c='cyan', s=50, alpha=0.3, label='test')
                plt.plot(x_test, o_out_, 'r-', lw=3, label='overfitting');
                plt.plot(x_test, d_out_, 'b--', lw=3, label='dropout(50%)')
                plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color': 'red'});
                plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
                plt.legend(loc='upper left');
                plt.ylim((-2.5, 2.5));
                plt.pause(0.1)

        plt.ioff()
        plt.show()
    pass

