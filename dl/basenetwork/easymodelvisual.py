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
with tf.name_scope("inputs"):
    x = tf.placeholder(tf.float32, [None, 1], name='x_in')
    y = tf.placeholder(tf.float32, [None, 1], name='y_in')

# 定义网络结构，这里定义两层
layer1 = LayerUtil.add_layer(x, 1, 10, n_layer=1, active_function=tf.nn.relu)
prediction = LayerUtil.add_layer(layer1, 10, 1, n_layer=2)
# 定义损失函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
    #设置loss的变化图标，loss的变化图标显示在scalar中
    tf.summary.scalar('loss', loss)
# 定义优化算法，这里采用梯度下降
with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化所有变量
init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        # tf.merge_all_summaries() 方法会把所有的 summaries 合并到一起
        merged = tf.summary.merge_all()
        '''
        tf.summary.FileWriter()可以将结果输出到一个文件目录中，第一个参数是目标目录，第二个参数需要使用sess.graph，
        因此需要把这句话放在获取session的后面。在目标文件目录的上一级目录中，在终端下输入：tensorboard --logdir 目录名，
        用tensorboard打开tensorboard服务，在谷歌浏览器中查看结果
        '''
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        for i in range(20001):
            sess.run(train, feed_dict={x: x_data, y: y_data})
            if i % 100 == 0:
                '''
                以上的配置仅仅可以绘制出训练的图表， 但是不会记录训练的数据，需要自己记录数据。
                需要通过运行merged才能记录数据
                '''
                rs = sess.run(merged, feed_dict={x: x_data, y: y_data})
                writer.add_summary(rs, i)
                # 每100轮刷新一次图形
                try:
                    ax.lines.remove(lines[0])
                except Exception:
                    pass
                prediction_value = sess.run(prediction, feed_dict={x: x_data})
                # 画出预测值
                lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
                plt.pause(0.5)
                print(i, sess.run(loss, feed_dict={x: x_data, y: y_data}))

    pass


