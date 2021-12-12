import random
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans


global MAX
MAX = 10000.0

global END
END = 0.000000001

def create_dataframe(file):
    data = pd.read_csv(file)
    data.rename(columns={'age': 'Age', 'sex': 'Gender', 'cp': 'Chest_pain', 'trestbps': 'Resting_blood_pressure',
                            'chol': 'Cholesterol', 'fbs': 'Fasting_blood_sugar',
                            'restecg': 'ECG_results', 'thalach': 'Maximum_heart_rate',
                            'exang': 'Exercise_induced_angina', 'oldpeak': 'ST_depression', 'ca': 'Major_vessels',
                            'thal': 'Thalassemia_types', 'target': 'Heart_disease', 'slope': 'ST_slope'}, inplace=True)
    print(data.head())
    print(data.info())
    return data


def initialize_U(data, cluster_num):
    U = []
    global MAX
    for i in range(0,len(data)):
        current = []
        sum = 0.0
        for j in range(0, cluster_num):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            sum += dummy
        for j in range(0, cluster_num):
            current[j] = current[j]/sum
        U.append(current)
    return U


def distance(point, center, i, j):
    if point.columns.size != len(center[0]):
        return -1
    dis = 0.0
    for k in range(0, point.columns.size):
        dis += (abs(point.iloc[i][k]-center[j][k]))**2
    return math.sqrt(dis)


def calcu_C(data, U, cluster_num, m):
    C = []
    for j in range(0, cluster_num):
        current = []
        c_value = 0.0
        for i in range(0,data.columns.size):
            ux = 0.0
            u = 0.0
            for k in range(0, len(data)):
                ux += (U[k][j]**m)*data.iloc[k][i]
                u += U[k][j]**m
                c_value = ux/u
            current.append(c_value)
        C.append(current)
    return C


def renew_U(data, C, m, cluster_num):
    U = []
    for i in range(0, len(data)):
        current = []
        cur = []
        for j in range(0, cluster_num):
            dummy = distance(data, C, i, j)
            current.append(dummy)
        for j in range(0, cluster_num):
            sum = 0.0
            for k in range(0, cluster_num):
                sum += (current[j]/current[k])**(2/(m-1))
            cur.append(1/sum)
        U.append(cur)
    return U


def end_condition(U, U_old):
    global END
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j]-U_old[i][j])>END:
                return False
    return True


def normalize_U(U):
    for i in range(0,len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U


def check(U_cluster, data):
    if U_cluster[0][0] != 1:
        for i in range(0,len(data)):
            k = U_cluster[i][0]
            U_cluster[i][0] = U_cluster[i][1]
            U_cluster[i][1] = k
    r = 0.0
    f = 0.0
    for i in range(0, len(data)):
        if  U_cluster[i][0] == data.iloc[i][data.columns.size-1]:
            r += 1
        else:
            f += 1
    rate = r/(r+f)
    return rate


def FCM(data, cluster_num, m):
    U = initialize_U(data, cluster_num)
    C = calcu_C(data, U, cluster_num, m)
    U_old = U
    U = renew_U(data, C, m, cluster_num)
    while (True):
        if end_condition(U,U_old):
            print('End cluster')
            break
        else:
            C = calcu_C(data, U, cluster_num, m)
            U_old = U
            U = renew_U(data, C, m, cluster_num)
    U_cluster = normalize_U(U)
    return U, C, U_cluster


def plot_all_features_against_target(X, y):
    n_columns = len(list(X.columns))
    n_plot_rows = (n_columns+1)//2
    plt.figure(figsize=(12,24))
    plt.title("Feature Histgrams", fontsize=25)
    index = 1
    for column in X.columns:
        plt.subplot(n_plot_rows, 2, index)
        sns.kdeplot(data=X, x=column, hue=y)
        index += 1
    return


def score_dataset_by_MLPC(X, y, mark):
    scaler = RobustScaler()
    X_sc = scaler.fit_transform(X=X, y=y)
    mlpc = MLPClassifier(random_state=42)
    mlpc_accuracy = cross_val_score(
        mlpc, X_sc, y, cv=5, scoring="accuracy"
    )
    print(f"The {mark} dataset score by MLPC:")
    print(f"Accuracy is {mlpc_accuracy.mean():.4}")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (mlpc_accuracy.mean(), mlpc_accuracy.std()))
    return mlpc_accuracy


def calcu_kind(U_cluster, data, cluster_num):
    X = data.copy()
    y = X.pop("Heart_disease")
    Xclst = X.copy()
    kind = []
    for i in range(0,len(data)):
        k = 0
        for j in range(0, cluster_num):
            if U_cluster[i][j] !=1:
                k += 1
            else:
                break
        kind.append(k)
    Xclst["Cluster"] = kind
    Xclst["Cluster"] = Xclst["Cluster"].astype("category")
    return X, y, Xclst

def baseline(data):
    X = data.copy()
    y = X.pop("Heart_disease")
    baseline = score_dataset_by_MLPC(X, y, "baseline")
    return baseline


def kmeans_score(data):
    kmeans = KMeans(n_clusters=3)
    X = data.copy()
    y = X.pop("Heart_disease")
    Xclst = X.copy()
    yclst = y.copy()
    Xcd = kmeans.fit_transform(Xclst)
    Xcd = pd.DataFrame(Xcd, columns=[f"Centroid_{i}" for i in range(Xcd.shape[1])])
    Xclst = Xclst.join(Xcd)
    kmeans_score = score_dataset_by_MLPC(Xclst, yclst, "+K means clustering")
    return kmeans_score


if __name__ == '__main__':
    data = create_dataframe('/Users/guiletong/Desktop/heart.csv')
    Xplot = data.copy()
    yplot = Xplot.pop("Heart_disease")
    plot_all_features_against_target(Xplot, yplot)
    cluster_num = 3
    m = 3.5
    U, C, U_cluster = FCM(data, cluster_num, m)
    X, y, Xclst = calcu_kind(U_cluster, data, cluster_num)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x="Age", y="Cholesterol", hue="Cluster", data=Xclst)
    Fuzzy = score_dataset_by_MLPC(Xclst, y, "+Fuzzy C-means clustering")
    plt.show()
    print(Fuzzy)
    print(baseline(data))
    print(kmeans_score(data))