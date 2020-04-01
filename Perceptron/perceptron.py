#coding=utf-8

'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
正确率：81.72%（二分类）
运行时长：78.6s
'''

import numpy as np
import time

def loadData(fileName):
    '''
    加载Mnist数据集
    :param fileName:要加载的数据集路径
    :return: list形式的数据集及标记
    '''
    print('start to read data')
    #存放数据及标记的list
    dataArr = []
    labelArr = []
    fr = open(fileName, 'r')
    for line in fr.readlines():
        # 对每一行数据按','进行切割，返回字段列表
        curLine = line.strip().split(',')

        #存放标签
        # Mnsit有 0-9 共 10 类标记，由于perceptron是二分类器，所以将>=5的作为1，<5为-1
        if int(curLine[0]) >= 5:
            labelArr.append(1)
        else:
            labelArr.append(-1)

        #存放数据
        #[int(num) for num in curLine[1:]]
        #[int(num)/255 for num in curLine[1:]] #将所有数据除以255归一化（非必须步骤，可以不归一化）
        dataArr.append([int(num)/255 for num in curLine[1:]])

    #返回data和label
    return dataArr, labelArr


def perceptron(dataArr, labelArr, iter=50):
    '''
    感知机训练过程
    :param dataArr:训练集的数据 (list)
    :param labelArr: 训练集的标签(list)
    :param iter: 迭代次数，默认50
    :return: 训练好的w和b
    '''
    print('start to trans')
    #将二维列表的数据转换成矩阵形式，方便运算
    #dataMat为matrix类型
    dataMat = np.mat(dataArr)
    #将标签转换成矩阵，之后转置（.T为转置）
    #转置是因为在运算中需要单独取label中的某一个元素，如果是1xN的矩阵的话，无法用label[i]的方式读取
    #对于只有1xN的label可以不转换成矩阵，直接label[i]即可，这里转换是为了格式上的统一
    labelMat = np.mat(labelArr).T
    #获取数据矩阵的大小，为m*n
    m, n = np.shape(dataMat)
    #创建初始权重w，初始值全为 0
    
    
    w = np.zeros((1, n))
    #初始化偏置 b 为 0
    b = 0
    #初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
    h = 0.0001

    #进行iter次迭代计算
    for k in range(iter):
        #对于每一个样本进行梯度下降
        #李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一进行一次梯度下降

        #在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了
        #此时使用随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降
        #两者的差异各有千秋，但较为常用的是随机梯度下降
        for i in range(m):
            #获取当前样本的向量
            xi = dataMat[i] # (1,784)
            #获取当前样本所对应的标签
            yi = labelMat[i] # (1,1)
            #判断是否是误分类样本
            #误分类样本特征：-yi(w*xi+b)>=0，详细可参考书中2.2.2小节
            #在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
            if -1 * yi * (w * xi.T + b) >= 0:
                #对于误分类样本，进行梯度下降，更新w和b
                w = w + h *  yi * xi
                b = b + h * yi
        #打印训练进度
        print('Round %d:%d training' % (k, iter))

    #返回训练完的w、b
    return w, b


def model_test(dataArr, labelArr, w, b):
    '''
    测试准确率
    :param dataArr:测试集
    :param labelArr: 测试集标签
    :param w: 训练获得的权重w
    :param b: 训练获得的偏置b
    :return: 正确率
    '''
    print('start to test')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T

    #获取测试数据集矩阵的大小
    m, n = np.shape(dataMat)
    #错误样本数计数
    errorCnt = 0
    #遍历所有测试样本
    for i in range(m):
        #获得单个样本向量
        xi = dataMat[i]
        #获得该样本标记
        yi = labelMat[i]
        #获得运算结果
        result = -1 * yi * (w * xi.T + b)
        #如果-yi(w*xi+b)>=0，说明该样本被误分类，错误样本数加一
        if result >= 0: 
            errorCnt += 1
    #正确率 = 1 - （样本分类错误数 / 样本总数）
    accruRate = 1 - (errorCnt / m)
    #返回正确率
    return accruRate


if __name__ == '__main__':
    #获取当前时间
    #在文末同样获取当前时间，两时间差即为程序运行时间
    start = time.time()

    #获取训练集及标签
    trainData, trainLabel = loadData('./Mnist/mnist_train.csv')
    #获取测试集及标签
    testData, testLabel = loadData('./Mnist/mnist_test.csv')

    #训练获得权重
    w, b = perceptron(trainData, trainLabel, iter = 30)
    #进行测试，获得正确率
    accruRate = model_test(testData, testLabel, w, b)

    #获取当前时间，作为结束时间
    end = time.time()
    #显示正确率
    print('accuracy rate is:', accruRate)
    #显示用时时长
    print('time span:', end - start)

