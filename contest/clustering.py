'''
@Author: Wu Yuhui
@Time  : 2019.05.1
'''
import os, sys
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import dbscan

def txt_to_array(filename:str) -> [np.array, np.array]:
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
    
    fp.close()
    ret_array = np.array(ret_array)
    return ret_array[:, 0:-1], ret_array[:, -1]

def list_to_txt(l:list, filename:str):
    fp = open(filename, "w+")
    for ele in l:
        if ele == 1:
            fp.write("2\n")
        else:
            fp.write("4\n")
    fp.close()

def preprocessing(data_array: np.array) -> np.array:
    '''
    todo
    '''
    return


if __name__ == "__main__":

    # Move to the directory of this files
    path = os.path.abspath(sys.argv[0])
    dir = os.path.split(path)[0]
    os.chdir(dir)

    # 1. Read in "breast.txt", convert it to a np.array
    data_array, data_labels = txt_to_array("./breast.txt")

    # 2. Preprocessing
    # preprocessing(data_array)

    # 3. Clustering
    # dbscan best: eps = 4.5, minpoints = [18,24], NMI 0.781253265689395
    # result_labels = dbscan.dbscan(data_array, 4.5, 20)
    
    
    # 4. Calculate NMI
    nmi = normalized_mutual_info_score(result_labels, data_labels)
    print("NMI is", nmi)

    # 5. PCA and see the result
    fig = plt.figure()
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data_array[:, 0:-1])
    data_labels_plot = fig.add_subplot(121, projection = '3d')
    data_labels_plot.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], s=5, c=data_labels)
    result_labels_plot = fig.add_subplot(122, projection = '3d')
    result_labels_plot.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], s=5, c=result_labels)
    plt.show()

    
    '''
    # 5. Write your result to "<student ID>.txt"
    list_to_txt(result_labels, "./2016300030023.txt")
    '''
    