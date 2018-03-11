# -*- coding: utf-8 -*-

import numpy as np
import time
import random

class LogisticRegression():

    def __int__(self, normalize=False):
        self.weights=[]

    def fit(self, train_x, train_y, maxIter, learning_rate=0.01, optimizeType='GD', batch_size=0):
        # 记录训练开始的时间
        startTime = time.time()
        # 计算样本的数量和样本的特征数量
        numSamples, numFeatures = np.shape(train_x)
        self.weights = np.zeros((numFeatures, 1))
        # print(np.shape(self.weights))
        # 通过梯度下降算法来计算最优权重
        if optimizeType == 'GD':
            self.gradDescent(train_x, train_y, maxIter, learning_rate)
        elif optimizeType == 'SGD':
            self.stoGradDescent(train_x, train_y, maxIter, learning_rate, batch_size)
        else:
            raise NameError('Not support optimize method type!')
        print('训练结束，耗时 %fs!' % (time.time() - startTime))


    #梯度下降
    def gradDescent(self, train_x, train_y, maxIter, learning_rate):
        for i in range(maxIter):
            output = self.sigmoid(train_x * self.weights)
            error = output - train_y
            # 梯度下降更新参数
            self.weights = self.weights - learning_rate * train_x.transpose() * error

    #随机梯度下降
    def stoGradDescent(self, train_x, train_y, maxIter, learning_rate, batch_size):
        numSamples, numFeatures = np.shape(train_x)
        for i in range(maxIter):
            #随机生成一个batch
            train_x_batch = np.ones((batch_size,numFeatures))
            train_y_batch = np.ones((batch_size,1))

            for j in range(batch_size):
                index = random.randint(0, numSamples-1)
                train_x_batch[j]=train_x[index]
                train_y_batch[j]=train_y[index]

            train_x_batch=np.mat(train_x_batch)
            train_y_batch=np.mat(train_y_batch)
            output = self.sigmoid(train_x_batch * self.weights)
            error = output - train_y_batch
            # 梯度下降更新参数
            self.weights = self.weights - learning_rate * train_x_batch.transpose() * error


    #预测函数
    def predict(self, predict_x, threshold=0.5):
        result = []
        if predict_x is None :
            return result
        numSamples, numFeatures = np.shape(predict_x)
        if  numSamples == 0  or numFeatures != len(self.weights):
            return result
        for i in range(numSamples):
            output = self.sigmoid(predict_x[i,:] * self.weights)
            if output>threshold :
                result.append(1)
            else:
                result.append(0)
        return result

    #测试模型的正确率
    def testCorrectScore(self, test_x=None, test_y=None, threshold=0.5):
        if test_x is None or test_y is None :
            return .0
        predicts=self.predict(test_x, threshold)
        if len(predicts)==0:
            return .0
        correctNum=0
        for i in range(len(predicts)):
            if predicts[i]==test_y[i]:
                correctNum=correctNum+1
        accuracy = float(correctNum) / len(predicts)
        return accuracy

    # 定义sigmod函数
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

