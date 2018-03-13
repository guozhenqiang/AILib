# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from dl.util.LayerUtil import LayerUtil
import matplotlib.pyplot as plt

# 构造训练集
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
#画出数据集
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
# 定义占位符
x = tf.placeholder(tf.float32, [None, 1], name='x_in')
y = tf.placeholder(tf.float32, [None, 1], name='y_in')

# 定义网络结构，这里定义两层
layer1 = LayerUtil.add_layer(x, 1, 10, n_layer=1, active_function=tf.nn.relu)
prediction = LayerUtil.add_layer(layer1, 10, 1, n_layer=2)
# 定义损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
# 定义优化算法，这里采用梯度下降
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所有变量
init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        for i in range(20001):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if i % 100 == 0:
                # 每100轮刷新一次图形
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={x: x_data})
                # plot the prediction
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.5)
                print(i, sess.run(loss, feed_dict={x: x_data, y: y_data}))

    pass
