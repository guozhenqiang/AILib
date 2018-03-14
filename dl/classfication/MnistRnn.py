# -*- coding: utf-8 -*-

'''
使用RNN模型来实现手写数字分类问题，RNN从每张图片的第一行像素读到最后一行，然后进行分类判断
'''
import tensorflow as tf
from dl.util.ModelEvaluationUtil import ModelEvaluationUtil
from tensorflow.examples.tutorials.mnist import input_data

#导入MNIST数据集
mnist = input_data.read_data_sets("D:\\tensorflow\\cha5\\mnist_data", one_hot=True)

# 定义模型中的超参数
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# 定义输入
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义weights和biases
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

#定义RNN网络结构
def RNN(X, weights, biases):
    # 原始的X是3维数据, 需要把它变成2维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batch * 28 steps , 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # 重新换回3维 X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # basic LSTM Cell.
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


prediction = RNN(x, weights, biases)
#定义损失函数和优化方法
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
train = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            print(batch_xs[1])
            print(batch_ys[1])
            exit(0)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train], feed_dict={x: batch_xs, y: batch_ys})
            if step % 20 == 0:
                print(step, ModelEvaluationUtil.compute_accuracy(sess, prediction, x, y, batch_xs, batch_ys))
            step += 1

