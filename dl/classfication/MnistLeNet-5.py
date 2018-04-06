# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#导入MNIST数据集
mnist = input_data.read_data_sets("D:\\tensorflow\\cha5\\mnist_data", one_hot=True)


input_node = 28*28
output_node = 10
image_size=28
channels_num = 1
batch_size = 20
conv1_deep = 32
conv1_size = 5

conv2_deep = 64
conv2_size = 5
fc_size = 512


# 定义LeNet-5网络结构，卷积层1-池化层1-卷积层2-池化层2-全连接层1-全连接层2
def inference(input_tensor, train, regularizer, keep_prob):
    # 卷积层1，28*28*1
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight', [conv1_size, conv1_size, channels_num, conv1_deep], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases', [conv1_deep], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 池化层1，28*28*32
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 卷积层2，14*14*32
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight', [conv2_size, conv2_size, conv1_deep, conv2_deep], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', [conv2_deep], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 池化层2，14*14*64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 7*7*64
    pool_shape = pool2.get_shape().as_list()
    # print(pool_shape)
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2, [-1, nodes])
    # 全连接层1，7*7*64，512
    with tf.variable_scope('layer5-fc1'):
        fc1_weight = tf.get_variable('weight', [nodes, fc_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None :
            tf.add_to_collection('losses', regularizer(fc1_weight))
        fc1_biases = tf.get_variable('biases', [fc_size], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weight)+fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob)

    # 全连接层2，512，10
    with tf.variable_scope('layer6-fc2'):
        fc2_weight = tf.get_variable('weight', [fc_size, output_node],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weight))
        fc2_biases = tf.get_variable('biases', [output_node], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weight) + fc2_biases
    # 返回网络的结果
    return fc2


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


xs = tf.placeholder(tf.float32, [None, 784], name='x-input')
ys = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(xs, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)

prediction = tf.nn.softmax(inference(x_image, True, None, keep_prob))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
init = tf.global_variables_initializer()


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
            if i % 50 == 0:
                print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
    pass

