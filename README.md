# Machine-Learning-Coding
《统计学习方法》学习与实现

## 代码
代码基于https://github.com/Dod-o/Statistical-Learning-Method_Code

## 数据集
MNIST数据集是手写数字数据集，包含 60,000 个训练样本和 10,000 个测试样本，0-9共10种手写数字。  
预处理后的.csv数据文件，每行的第一个元素是数据的label，之后的28\*28个元素取值范围在\[0, 255\]，表示像素点的颜色。

## 生成式模型
生成式模型由数据学习联合概率分布P(X,Y)，然后求出条件概率分布P(Y|X)作为预测的模型，即：P(X|Y)=P(X,Y)/P(X)。  
典型的生成式模型有：朴素贝叶斯、隐马尔可夫模型。  

朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。  

## 判别式模型
判别式模型由数据直接学习决策函数f(X)或者条件概率分布P(Y|X)作为预测的模型。  
典型的判别式模型有：k近邻法、感知机、决策树、逻辑斯谛回归模型、最大熵模型、支持向量机、提升方法、条件随机场。  

k近邻方法是一种基本分类与回归方法。  
感知机是二类分类的线性分类模型。  
决策树是一种基本的分类与回归方法。  


