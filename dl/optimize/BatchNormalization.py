# -*- coding: utf-8 -*-

import tensorflow as tf
from util.LayerUtil import LayerUtil
import matplotlib.pyplot as plt
import numpy as np


active_function = tf.nn.relu
hidden_layer_num = 7
hidden_layer_units = 30
learning_rate = 0.001


def fix_seed(seed=1):
    # 初始化所有的随机函数的种子
    np.random.seed(seed)
    tf.set_random_seed(seed)


def plot_his(inputs, inputs_norm):
    # plot histogram for the inputs of every layer
    for j, all_inputs in enumerate([inputs, inputs_norm]):
        for i, input in enumerate(all_inputs):
            plt.subplot(2, len(all_inputs), j*len(all_inputs)+(i+1))
            plt.cla()
            if i == 0:
                the_range = (-7, 10)
            else:
                the_range = (-1, 1)
            plt.hist(input.ravel(), bins=15, range=the_range, color='#FF5733')
            plt.yticks(())
            if j == 1:
                plt.xticks(the_range)
            else:
                plt.xticks(())
            ax = plt.gca()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
        plt.title("%s normalizing" % ("Without" if j == 0 else "With"))
    plt.draw()
    plt.pause(0.01)


def add_layer(inputs, in_size, out_size, active_function=None, norm=False):
    # weights and biases (bad initialization for this case)
    Weights = tf.Variable(tf.random_normal([in_size, out_size], mean=0., stddev=1.))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    # fully connected product
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # normalize fully connected product
    if norm:
        # Batch Normalize
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0],  # the dimension you wanna normalize, here [0] for batch
            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        )
        scale = tf.Variable(tf.ones([out_size]))
        shift = tf.Variable(tf.zeros([out_size]))
        epsilon = 0.001

        # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()

        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        # 标准化需要两步:
        # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
        # Wx_plus_b = Wx_plus_b * scale + shift

    # 激活函数
    if active_function is None:
        outputs = Wx_plus_b
    else:
        outputs = active_function(Wx_plus_b)

    return outputs



def built_net(xs, ys, norm):

    fix_seed(1)
    layers_inputs = [xs]    # 记录每层的 input

    # loop 建立所有层
    for l_n in range(hidden_layer_num):
        layer_input = layers_inputs[l_n]
        in_size = layers_inputs[l_n].get_shape()[1].value
        output = add_layer(layer_input,in_size, hidden_layer_units, active_function, norm)
        layers_inputs.append(output)    # 把 output 加入记录

    # 建立 output layer
    prediction = add_layer(layers_inputs[-1], 30, 1, None)

    cost = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return [train_op, cost, layers_inputs]


# 构造数据集
fix_seed(1)
x_data = np.linspace(-7, 10, 2500)[:, np.newaxis]
np.random.shuffle(x_data)
noise = np.random.normal(0, 8, x_data.shape)
y_data = np.square(x_data) - 5 + noise

# 可视化数据集
# plt.scatter(x_data, y_data)
# plt.show()

xs = tf.placeholder(tf.float32, [None, 1])  # [num_samples, num_features]
ys = tf.placeholder(tf.float32, [None, 1])

train_op, cost, layers_inputs = built_net(xs, ys, norm=False)   # without BN
train_op_norm, cost_norm, layers_inputs_norm = built_net(xs, ys, norm=True)  # with BN

init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        # record cost
        cost_his = []
        cost_his_norm = []
        record_step = 5
        plt.ion()
        plt.figure(figsize=(7, 3))
        for i in range(250):
            if i % 50 == 0:
                # plot histogram
                all_inputs, all_inputs_norm = sess.run([layers_inputs, layers_inputs_norm],
                                                       feed_dict={xs: x_data, ys: y_data})
                plot_his(all_inputs, all_inputs_norm)

            # train on batch
            sess.run([train_op, train_op_norm],
                     feed_dict={xs: x_data[i * 10:i * 10 + 10], ys: y_data[i * 10:i * 10 + 10]})

            if i % record_step == 0:
                # record cost
                cost_his.append(sess.run(cost, feed_dict={xs: x_data, ys: y_data}))
                cost_his_norm.append(sess.run(cost_norm, feed_dict={xs: x_data, ys: y_data}))

        plt.ioff()
        plt.figure()
        plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his), label='no BN')  # no norm
        plt.plot(np.arange(len(cost_his)) * record_step, np.array(cost_his_norm), label='BN')  # norm
        plt.legend()
        plt.show()

    pass

