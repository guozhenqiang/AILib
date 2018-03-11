# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from model import LogisticRegression
from model import LinearRegression

def loadData():
    train_x = []
    train_y = []
    fileIn = open('E:/testSet.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return np.mat(train_x), np.mat(train_y).transpose()

def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = np.shape(train_x)
    if numFeatures != 3:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

        # draw all samples
    for i in range(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

            # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.show()

if __name__ == '__main__':

        ## step 1: load data
        print("step 1: load data...")
        train_x, train_y = loadData()

        test_x = train_x
        test_y = train_y

        ## step 2: training...
        print( "step 2: training...")

        LogRegModel=LogisticRegression.LogisticRegression()
        # LogRegModel=LinearRegression.LinearRegression()
        LogRegModel.fit(train_x, train_y, 10000, 0.01, 'SGD', 20)

        ## step 3: testing
        print("step 3: testing...")
        accuracy = LogRegModel.testCorrectScore(test_x, test_y, 0.5)

        ## step 4: show the result
        print("step 4: show the result...")
        print('分类准确率: %.3f%%' % (accuracy * 100))
        showLogRegres(LogRegModel.weights, train_x, train_y)
