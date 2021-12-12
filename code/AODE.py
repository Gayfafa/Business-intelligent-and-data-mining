from pylab import *
import numpy as np

featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']}


def getDataSet():
    dataset = [
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, 1],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, 1],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, 1],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, 1],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, 1],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, 1],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, 1],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, 0],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, 0],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, 0],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, 0],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, 0],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, 0],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, 0],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, 0]
    ]

    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量']
    numlist = []
    for i in range(len(features) - 2):
        numlist.append(len(featureDic[features[i]]))

    dataset = np.array(dataset)
    return dataset, features


def AODE(dataset, data, features):
    m, n = dataset.shape
    n = n - 3
    pDir = {}
    for classLabel in ["好瓜", "坏瓜"]:
        P = 0.0
        if classLabel == "好瓜":
            sign = '1'
        else:
            sign = '0'
        extrDataSet = dataset[dataset[:, -1] == sign]
        for i in range(n):
            xi = data[i]
            Dcxi = extrDataSet[extrDataSet[:, i] == xi]
            Ni = len(featureDic[features[i]])
            Pcxi = (len(Dcxi) + 1) / float(m + 2 * Ni)
            mulPCond = 1
            for j in range(n):
                xj = data[j]
                Dcxij = Dcxi[Dcxi[:, j] == xj]
                Nj = len(featureDic[features[j]])
                PCond = (len(Dcxij) + 1) / float(len(Dcxi) + Nj)
                mulPCond *= PCond
            P += Pcxi * mulPCond
        pDir[classLabel] = P

    if pDir["好瓜"] > pDir["坏瓜"]:
        preClass = "好瓜"
    else:
        preClass = "坏瓜"

    return pDir["好瓜"], pDir["坏瓜"], preClass


def calcAccRate(dataset, features):
    cnt = 0
    for data in dataset:
        _, _, pre = AODE(dataset, data, features)
        if (pre == '好瓜' and data[-1] == '1') \
            or (pre == '坏瓜' and data[-1] == '0'):
            cnt += 1
    return cnt / float(len(dataset))


def main():
    dataset, features = getDataSet()
    data = np.array(['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, 1])
    pg, pb, pre = AODE(dataset, data, features)
    print("pG = ", pg)
    print("pB = ", pb)
    print("pre = ", pre)
    print(calcAccRate(dataset, features))


if __name__ == '__main__':
    main()

