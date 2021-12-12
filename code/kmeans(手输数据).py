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

def createDataSet():
    return [[0.697,0.46],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],
            [0.403,0.237],[0.481,0.149],[0.437,0.211],[0.666,0.091],[0.243,0.267],
            [0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],[0.36,0.37],
            [0.593,0.042],[0.719,0.103],[0.359,0.188],[0.339,0.241],[0.282,0.257],
            [0.748,0.232],[0.714,0.346],[0.483,0.312],[0.478,0.437],[0.525,0.369],
            [0.751,0.489],[0.532,0.472],[0.473,0.376],[0.725,0.445],[0.446,0.459]]

if __name__ == '__main__':
    dataset = createDataSet()
    centroids, cluster = kmeans(dataset, 7)
    print('质心为：%s' % centroids)
    print('集群为：%s' % cluster)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1], marker='o', color='green', s=40, label='原始点')
        for j in range(len(centroids)):
            plt.scatter(centroids[j][0], centroids[j][1], marker='x', color='red', s=50, label='质心')
    plt.show()

