# coding:utf8
import math
import numpy as np

def sigmod(x):
    return 1.0 / (1.0 + math.exp(x))


def gradAscent(data, label):
    dataMatrix = np.mat(data)
    labelMatrix = np.mat(label).transpose()

    # 矩阵的行数和列数
    m, n = np.shape(dataMatrix)

    # 学习率
    alpha = 0.01

    # 迭代次数
    max_iter = 500

    # 初始化权重为1
    weights = np.ones((n, 1))

    for k in range(max_iter):
        h = sigmod(dataMatrix * weights)
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights