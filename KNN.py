# -*- coding: utf-8 -*-
__author__ = 'cm'
#date : 2018/3/19

#KNN紧邻算法2.3 准备数据：归一化数值
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#inX 分类的输入向量
#dataSet 训练样本集
#labels 标签向量
#k 表示用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #shape查看矩阵维数 shape[0]表示矩阵行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #①距离计算
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #加入参数axis=1以后就是将一个矩阵的每一行向量相加
    distances = sqDistances**0.5 #例子中计算出的值 array([ 1.48660687, 1.41421356, 0. , 0.1 ])
    sortedDistIndicies = distances.argsort() #argsort()方法得到矩阵中每个元素的排序序号 array([2, 3, 1, 0])
    classCount={}
    for i in range(k): #选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #k=3 最后classCount的值为 {'A': 1, 'B': 2}
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #③排序
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # tmp = createDataSet()
    # print tmp


