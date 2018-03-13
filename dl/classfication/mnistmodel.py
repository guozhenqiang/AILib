# -*- coding: utf-8 -*-

import tensorflow as tf
from dl.util.LayerUtil import LayerUtil
from dl.util.ModelEvaluationUtil import ModelEvaluationUtil
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:\\tensorflow\\cha5\\mnist_data", one_hot=True)
# 定义占位符
x = tf.placeholder(tf.float32, [None, 784], name='x_in')
y = tf.placeholder(tf.float32, [None, 10], name='y_in')



# 定义网络结构，这里定义两层
# layer1 = LayerUtil.add_layer(x, 784, 100, n_layer=1, active_function=tf.nn.relu)
# prediction = LayerUtil.add_layer(layer1, 100, 10, n_layer=2, active_function=tf.nn.softmax)
prediction = LayerUtil.add_layer(x, 784, 10, n_layer=1, active_function=tf.nn.softmax)
# 定义损失函数
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
# 定义优化算法，这里采用梯度下降
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# 初始化所有变量
init = tf.global_variables_initializer()


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        batch_size = 100
        for i in range(1001):
            # 从train集合中选取batch_size个训练数据
            x_data, y_data = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if i % 50 == 0:
                print(ModelEvaluationUtil.compute_accuracy(sess, prediction, x, y, mnist.test.images, mnist.test.labels))

    pass

