import os, sys
from numpy import *

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def csv2matrix(filename):
    '''
    
    '''

    fp = open(filename)
    line = fp.readline()
    a = []
    b = []
    c = []
    while line:
        line = line.strip('\n')
        line = line.split(',')
        if line[-1] == 'A':
            a.append(line[0:3])
        elif line[-1] == 'B':
            b.append(line[0:3])
        elif line[-1] == 'C':
            c.append(line[0:3])
        line = fp.readline()
    
    ret_matrix_a = zeros((3, len(a)))
    ret_matrix_b = zeros((3, len(b)))
    ret_matrix_c = zeros((3, len(c)))

    for i in range(len(a)):
        ret_matrix_a[:,i] = a[i]
    for i in range(len(b)):
        ret_matrix_b[:,i] = b[i]
    for i in range(len(c)):
        ret_matrix_c[:,i] = c[i]

    return ret_matrix_a, ret_matrix_b, ret_matrix_c


def pca(data_matrix, dim):
    '''
    输出: 投影矩阵 ret_matrix
    '''

    # 1. 中心化, data_matrix -> X
    X = zeros((size(data_matrix, 0), size(data_matrix, 1)))
    
    center = zeros((1, size(data_matrix, 0)))
    
    for i in range(size(data_matrix, 1)):
        center += data_matrix[:,i]
    center /= size(data_matrix, 1)

    for i in range(size(data_matrix, 1)):
        X[:,i] = data_matrix[:,i] - center
    
    # 2. 计算 X 的协方差矩阵 Cov
    Cov = X.dot(X.T)
    Cov = Cov.real # 去掉虚部

    # 3. 对 Cov 进行特征值分解
    w,v = linalg.eig(Cov)

    # 4. 对分解出的特征值逆序排序, 取前 dim 项对应的特征向量返回
    w_sorted = sort(w)[::-1]
    
    ret_matrix = zeros((dim,size(v, 1)))
    for i in range(dim):
        for j in range(size(w, 0)):
            if w_sorted[i] == w[j]:
                ret_matrix[i,:] = v[j,:]

    return ret_matrix

if __name__ == '__main__':
    
    path = os.path.abspath(sys.argv[0])
    dir = os.path.split(path)[0]
    os.chdir(dir)

    data_matrix_a, data_matrix_b, data_matrix_c = csv2matrix('./dataset_pca.csv')
    data_matrix_all = append(data_matrix_a, data_matrix_b, axis = 1)
    data_matrix_all = append(data_matrix_all, data_matrix_c, axis = 1)

    proj_matrix_2d = pca(data_matrix_all, 2)

    result_matrix_2d_a = proj_matrix_2d.dot(data_matrix_a)
    result_matrix_2d_b = proj_matrix_2d.dot(data_matrix_b)
    result_matrix_2d_c = proj_matrix_2d.dot(data_matrix_c)

    result_matrix_1d_a = proj_matrix_2d[0,:].dot(data_matrix_a)
    result_matrix_1d_b = proj_matrix_2d[0,:].dot(data_matrix_b)
    result_matrix_1d_c = proj_matrix_2d[0,:].dot(data_matrix_c)

    
    # show figures
    fig = plt.figure()
    ax_original = fig.add_subplot(131, projection = '3d')
    ax_original.scatter(data_matrix_a[0,:], data_matrix_a[1,:], data_matrix_a[2,:], s=5, c='red')
    ax_original.scatter(data_matrix_b[0,:], data_matrix_b[1,:], data_matrix_b[2,:], s=5, c='blue')
    ax_original.scatter(data_matrix_c[0,:], data_matrix_c[1,:], data_matrix_c[2,:], s=5, c='yellow')

    ax_2d = fig.add_subplot(132)
    ax_2d.scatter(result_matrix_2d_a[0,:], result_matrix_2d_a[1,:], s=5, c='red')
    ax_2d.scatter(result_matrix_2d_b[0,:], result_matrix_2d_b[1,:], s=5, c='blue')
    ax_2d.scatter(result_matrix_2d_c[0,:], result_matrix_2d_c[1,:], s=5, c='yellow')

    ax_1d = fig.add_subplot(133)
    
    ax_1d.scatter(result_matrix_1d_a[:], len(result_matrix_1d_a) * [0], s=5, c='red')
    ax_1d.scatter(result_matrix_1d_b[:], len(result_matrix_1d_b) * [0], s=5, c='blue')
    ax_1d.scatter(result_matrix_1d_c[:], len(result_matrix_1d_c) * [0], s=5, c='yellow')
    
    plt.show()
