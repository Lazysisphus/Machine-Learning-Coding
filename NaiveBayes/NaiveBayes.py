# coding=utf-8


'''
数据集：Mnist
训练集数量：60000
测试集数量：10000
------------------------------
运行结果：
    正确率：84.3%
    运行时长：103s
'''

import numpy as np
import time


def loadData(fileName):
    '''
    加载文件
    :param fileName: 要加载的文件路径
    :return: 数据集和标签集
    '''
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # 由于后续需要计算条件概率 P(X|Y)，即标签为Y条件下某一维度取值为X的概率
        # 如果X的可取值过多，不方便条件概率的计算
        # 此外将数据进行了二值化处理，大于128的转换成1，小于等于的转换成0，方便后续计算
        dataArr.append([int(int(num) > 128) for num in curLine[1:]])
        labelArr.append(int(curLine[0]))
    return dataArr, labelArr


def getAllProbability(trainDataArr, trainLabelArr):
    '''
    通过训练集计算先验概率分布和条件概率分布，是NB的主要计算过程
    :param trainDataArr: 训练数据集，二维列表，60000*(28*28)
    :param trainLabelArr: 训练标记集，一维列表，len==60000
    :return: 先验概率分布和条件概率分布
    '''
    # 设置数据的 特征数目 及 类别数目
    featureNum = 784
    classNum = 10

    # 计算先验概率
    # 初始化先验概率分布存放矩阵，计算得到的P(Y=0)放在Py[0]中，以此类推
    Py = np.zeros((classNum, 1))
    # 计算每个类别的先验概率分布
    # 使用 贝叶斯估计 及 拉普拉斯平滑 估算概率参数
    for i in range(classNum):
        # np.mat(trainLabelArr): Matrix类型，shape为60000*1
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)
    # 转换为log对数形式
    # log书中没有写到，但是实际中需要考虑到，原因是这样：
    # 最后求后验概率估计的时候，形式是各项的相乘（式4.7）
    # 这里存在两个问题：
    # 1.某一项为0时，结果为0，这个问题通过分子和分母加上一个相应的数可以排除，前面已经做好了处理
    # 2.如果特征特别多（例如在这里，需要连乘的项目有784个特征，加一个先验概率分布一共795项相乘，所有数都是0-1之间，结果一定是一个很小的接近0的数。），
    #   理论上可以通过结果的大小值判断， 但在程序运行中很可能会向下溢出无法比较，因为值太小了，所以人为把值进行log处理，
    #   log在定义域内是一个递增函数，也就是说log（x）中，x越大，log也就越大，单调性和原数据保持一致，
    #   所以加上log对结果没有影响，此外连乘项通过log以后，可以变成各项累加，简化了计算。
    Py = np.log(Py)

    # 计算条件概率
    # 初始化条件概率存放矩阵Px_y=P(X=x|Y=y)
    # 下面循环的直观含义是：如果某样本x的标签为y，且其在第j个特征的取值为x[j]，那么令Px_y[y][j][x[j]]加1
    # 概括地讲，就是统计每种标签条件下，每个特征的每个取值的出现次数
    Px_y = np.zeros((classNum, featureNum, 2))
    for i in range(len(trainLabelArr)):
        # 获取当前循环所使用的标记
        label = trainLabelArr[i]
        # 获取当前要处理的样本
        x = trainDataArr[i]
        # 对该样本的每一维特诊进行遍历
        for j in range(featureNum):
            # 在矩阵中对应位置加1
            # 这里还没有计算条件概率，先把所有数累加，全加完以后，在后续步骤中再求对应的条件概率
            Px_y[label][j][x[j]] += 1

    # 计算朴素贝叶斯公式的分母，以及分子和分母之间的除法
    for label in range(classNum):
        for j in range(featureNum):
            # 获取y=label，第j个特诊为0的个数
            Px_y0 = Px_y[label][j][0]
            # 获取y=label，第j个特诊为1的个数
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    # 返回 先验概率 和 条件概率
    return Py, Px_y


def NaiveBayes(Py, Px_y, x):
    '''
    通过朴素贝叶斯估计样本x对应每个标签的概率
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 要估计的样本x
    :return: 返回所有label的估计概率
    '''
    featrueNum = 784
    classNum = 10
    # 建立存放所有标记的估计概率数组
    P = [0] * classNum
    # 对于每一个类别，单独估计其概率
    for i in range(classNum):
        # 初始化sum为0，sum为求和项
        # 在训练过程中对概率进行了log处理，所以这里原先应当是连乘所有概率，最后比较哪个概率最大
        # 但是当使用log处理时，连乘变成了累加，所以使用sum
        sum = 0
        # 获取每一个条件概率值，进行累加
        for j in range(featrueNum):
            sum += Px_y[i][j][x[j]]
        # 最后再和先验概率相加
        P[i] = sum + Py[i]

    # max(P)：找到概率最大值
    # P.index(max(P))：找到该概率最大值对应的所有（索引值和标签值相等）
    return P.index(max(P))


def model_test(Py, Px_y, testDataArr, testLabelArr):
    '''
    对测试集进行测试
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param testDataArr: 测试集数据
    :param testLabelArr: 测试集标记
    :return: 准确率
    '''
    # 错误值计数
    errorCnt = 0
    # 循环遍历测试集中的每一个样本
    for i in range(len(testDataArr)):
        # 获取预测值
        presict = NaiveBayes(Py, Px_y, testDataArr[i])
        # 与答案进行比较
        if presict != testLabelArr[i]:
            # 若错误  错误值计数加1
            errorCnt += 1
    # 返回准确率
    return 1 - (errorCnt / len(testDataArr))


if __name__ == "__main__":
    start = time.time()
    # 获取训练集
    print('start read transSet')
    trainDataArr, trainLabelArr = loadData('./Mnist/mnist_train.csv')

    # 获取测试集
    print('start read testSet')
    testDataArr, testLabelArr = loadData('./Mnist/mnist_test.csv')

    # 开始训练，学习 先验概率 和 条件概率
    print('start to train')
    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)

    # 使用习得的 先验概率 和 条件概率 对测试集进行测试
    print('start to test')
    accuracy = model_test(Py, Px_y, testDataArr, testLabelArr)

    # 准确率
    print('the accuracy is:', accuracy)
    # 时间
    print('time span:', time.time() -start)
