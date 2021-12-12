import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k,1)) - centroids
        squaredDiff = diff ** 2
        squaredDist = np.sum(squaredDiff, axis=1)
        distance = squaredDist ** 0.5  # 开根号
        clalist.append(distance)
    clalist = np.array(clalist)
    return clalist

def classify(dataSet, centroids, k):
    clalist = calcDis(dataSet, centroids, k)
    minDistIndices = np.argmin(clalist, axis=1)
    newCentroids = pd.DataFrame(dataSet).groupby(
        minDistIndices).mean()
    newCentroids = newCentroids.values
    changed = newCentroids - centroids
    return changed, newCentroids

def kmeans(dataSet, k):
    centroids = random.sample(dataSet, k)
    changed, newCentroids = classify(dataSet, centroids, k)
    while np.any(changed != 0):
        changed, newCentroids = classify(dataSet, newCentroids, k)
    centroids = sorted(newCentroids.tolist())
    cluster = []
    clalist = calcDis(dataSet, centroids, k)
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):
        cluster[j].append(dataSet[i])

    return centroids, cluster

def loadxle(dir):
    data = pd.read_excel(dir)
    del data['编号']
    data = np.mat(data)
    data = data.tolist()
    return data

if __name__ == '__main__':
    dataset = loadxle('/Users/guiletong/Desktop/大三下学期/商务智能与数据挖掘/watermelon.xlsx')
    centroids, cluster = kmeans(dataset, 5)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
        for j in range(len(centroids)):
            plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()