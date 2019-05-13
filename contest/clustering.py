import os, sys
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

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


if __name__ == "__main__":

    # Move to the directory of this files
    path = os.path.abspath(sys.argv[0])
    dir = os.path.split(path)[0]
    os.chdir(dir)

    # 1. 读入 breast.txt, 转成矩阵
    data_array, data_labels = txt_to_array("./breast.txt")

    '''
    # 2. 对 dataset 聚类, 返回 1*n 类别矩阵
    result = clustering(data_array)
    '''

    # 3. 调库计算 NMI, 开奖
    nmi = normalized_mutual_info_score(result, data_labels)
    print("NMI is", nmi)
    
