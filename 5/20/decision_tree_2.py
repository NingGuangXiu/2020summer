import math
import operator
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ["SimHei"]
mpl.rcParams['axes.unicode_minus'] = True

def createDataSet():
    dataSet = [[]]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    #特征
    return dataSet,labels

dataset,dataLabels = createDataSet()

#计算信息熵
def calcShannonEnt(dataSet):
    #样本总个数
    totalNum = len(dataSet)
    #类别集合
    labelSet = {}
    #计算每个类别的样本个数
    for dataVec in dataSet:
        label = dataVec[-1]
        if label not in labelSet.keys():
            labelSet[label] = 0
        labelSet[label] += 1
    shannonEnt = 0
    #计算熵值
    for key in labelSet:
        pi = float(labelSet[key])/totalNum
        shannonEnt -= pi*math.log(pi,2)
    return shannonEnt

#按给定特征划分数据集:返回第featNum个特征其值为value的样本集合，且返回的样本数据中已经去除该特征
def splitDataSet(dataSet, featNum, featvalue):
    retDataSet = []
    for dataVec in dataSet:
        if dataVec[featNum] == featvalue:
            splitData = dataVec[:featNum]
            splitData.extend(dataVec[featNum+1:])
            retDataSet.append(splitData)
    return retDataSet

#选择最好的特征划分数据集
def chooseBestFeatToSplit(dataSet):
    featNum = len(dataSet[0]) - 1
    maxInfoGain = 0
    bestFeat = -1
    #计算样本熵值，对应公式中：H(X)
    baseShanno = calcShannonEnt(dataSet)
    #以每一个特征进行分类，找出使信息增益最大的特征
    for i in range(featNum):
        featList = [dataVec[i] for dataVec in dataSet]
        featList = set(featList)
        newShanno = 0
        #计算以第i个特征进行分类后的熵值，对应公式中：H(X|Y)
        for featValue in featList:
            subDataSet = splitDataSet(dataSet, i, featValue)
            prob = len(subDataSet)/float(len(dataSet))
            newShanno += prob*calcShannonEnt(subDataSet)
        #ID3算法：计算信息增益,对应公式中：g(X,Y)=H(X)-H(X|Y)
        infoGain = baseShanno - newShanno

        #找出最大的熵值以及其对应的特征
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            bestFeat = i
        print (bestFeat)
    return bestFeat

