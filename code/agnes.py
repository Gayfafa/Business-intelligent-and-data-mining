import numpy as np
import matplotlib.pyplot as plt

data = '''
1,0.697,0.460,
2,0.774,0.376,
3,0.634,0.264,
4,0.608,0.318,
5,0.556,0.215,
6,0.403,0.237,
7,0.481,0.149,
8,0.437,0.211,
9,0.666,0.091,
10,0.243,0.267,
11,0.245,0.057,
12,0.343,0.099,
13,0.639,0.161,
14,0.657,0.198,
15,0.360,0.370,
16,0.593,0.042,
17,0.719,0.103,
18,0.359,0.188,
19,0.339,0.241,
20,0.282,0.257,
21,0.748,0.232,
22,0.714,0.346,
23,0.483,0.312,
24,0.478,0.437,
25,0.525,0.369,
26,0.751,0.489,
27,0.532,0.472,
28,0.473,0.376,
29,0.725,0.445,
30,0.446,0.459'''


def load_dataset(data):
    data_   = data.strip().split(',')
    dataset = [(float(data_[i]), float(data_[i+1])) for i in range(1, len(data_)-1, 3)]
    return dataset



def show_dataset(dataset):
    for item in dataset:
        plt.plot(item[0], item[1], 'ob')
    plt.title("Dataset")
    plt.show()

# 计算两点之间的欧氏距离并返回

def elu_distance(a, b):
    dist = np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    return dist

# 计算集合Ci, Cj间最小距离

def dist_min(ci, cj):
    return min(elu_distance(i, j) for i in ci for j in cj)

# 计算集合Ci, Cj间最大距离

def dist_max(ci, cj):
    return max(elu_distance(i, j) for i in ci for j in cj)

# 计算集合Ci, Cj间平均距离

def dist_avg(ci, cj):
    return sum(elu_distance(i, j) for i in ci for j in cj) / (len(ci) * len(cj))

# 找出距离最小的两个簇并

def find_index(m):
    min_dist = float('inf')
    x = y = 0
    for i in range(len(m)):
        for j in range(len(m[i])):
            if i != j and m[i][j] < min_dist:
                min_dist, x, y = m[i][j], i, j
    return x, y, min_dist


def agnes(dataset, dist, k):
    c, m = [], []
    for item in dataset:
        ci = []
        ci.append(item)
        c.append(ci)
    for i in c:
        mi = []
        for j in c:
            mi.append(dist(i, j))
        m.append(mi)
    q = len(dataset)

    # agnes
    while q > k:
        x, y, min_dist = find_index(m)
        c[x].extend(c[y])
        c.remove(c[y])
        m = []
        for i in c:
            mi = []
            for j in c:
                mi.append(dist(i, j))
            m.append(mi)
        q -= 1
    return c


def show_cluster(cluster):
    colors = ['or', 'ob', 'og', 'ok', 'oy', 'ow']
    for i in range(len(cluster)):
        for item in cluster[i]:
            plt.plot(item[0], item[1], colors[i])
    plt.title("AGNES Clustering")
    plt.show()




if __name__ == "__main__":
    dataset = load_dataset(data)
    show_dataset(dataset)
    k = 4
    cluster = agnes(dataset, dist_avg, k)
    show_cluster(cluster)
    print (cluster)