# -*- coding: utf-8 -*-
import tensorflow as tf


class ModelEvaluationUtil():

    def __init__(self):
        pass

    #分类问题计算模型的效果
    @staticmethod
    def compute_accuracy(sess, prediction, x, y, v_xs, v_ys):
        y_pre = sess.run(prediction, feed_dict={x: v_xs})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={x: v_xs, y: v_ys})
        return result

if __name__ == '__main__':
    pass

