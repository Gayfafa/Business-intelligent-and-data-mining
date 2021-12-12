import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def lvq(data , n_clusters , learn_rate=0.1):
    orginal = random.sample(data, n_clusters)
    for _ in range(30):
        sample = random.sample(data,1)
        sample = np.asarray(sample)
        orginal = np.asarray(orginal)
        min = np.argmin([np.linalg.norm(sample-orginal[i])for i in range(n_clusters)])
        for j in range(n_clusters):
            if j == min:
                orginal[j] = orginal[j] + (sample - orginal[j]) * learn_rate
            else:
                orginal[j] = orginal[j] - (sample - orginal[j]) * learn_rate
        orginal = orginal.tolist()

    return orginal

def loadxle (dir) :
    data = pd.read_excel(dir)
    del data['编号']
    data = np.mat(data)
    data = data.tolist()
    return data

if __name__ == '__main__':
    data = loadxle('/Users/guiletong/Desktop/大三下学期/商务智能与数据挖掘/watermelon.xlsx')
    LVQ = lvq(data, 5)
    print(LVQ)
    LVQ = np.asarray(LVQ)
    data = np.asarray(data)
    for j in range(len(data)):
        plt.scatter(data[j][0], data[j][1], marker='o', color='green', s=40)
    for i in range(len(LVQ)):
        plt.scatter(LVQ[i][0], LVQ[i][1], marker='x', color='red', s=40)
    plt.show()
