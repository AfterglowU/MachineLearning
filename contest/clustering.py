import os, sys
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import dbscan

def txt_to_array(filename: str) -> [np.array, np.array]:
    fp = open(filename)
    array_of_lines = fp.readlines()
    ret_array = []
    for line in array_of_lines:
        line = line.strip("\n")
        ret_array_row = []
        for data in line.split():
            frac, exp = data.split("e+")
            frac = int(float(frac))
            exp = int(exp)
            ret_array_row.append(frac*10**exp)
        ret_array.append(ret_array_row)
    
    ret_array = np.array(ret_array)
    return ret_array[:, 0:-1], ret_array[:, -1]


def pretreatment(data_array: np.array) -> np.array:
    attr_var = []
    for i in range(len(data_array[0])):
        attr_var.append(np.var(data_array[:, i]))
    attr_var = np.array(attr_var)
    return 


if __name__ == "__main__":

    # Move to the directory of this files
    path = os.path.abspath(sys.argv[0])
    dir = os.path.split(path)[0]
    os.chdir(dir)

    # 1. 读入 breast.txt, 转成矩阵
    data_array, data_labels = txt_to_array("./breast.txt")

    # 2.1. 预处理
    # pretreatment(data_array)

    # 2.2. 聚类
    result_labels = dbscan.dbscan(data_array, 4.5, 50)
    
    # 3. 调库计算 NMI
    nmi = normalized_mutual_info_score(result_labels, data_labels)
    print("NMI is", nmi)

    # 4. PCA 后看一下图像
    fig = plt.figure()
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_array[:, 0:-1])
    data_labels_plot = fig.add_subplot(121, projection = '3d')
    data_labels_plot.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], s=5, c=data_labels)
    result_labels_plot = fig.add_subplot(122, projection = '3d')
    result_labels_plot.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], s=5, c=result_labels)
    plt.show()
