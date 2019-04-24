import os, sys
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

import sampling

def csv2matrix(filename):
    fp = open(filename)
    array_of_lines = fp.readlines()
    number_of_lines = len(array_of_lines)
    ret_matrix = zeros((number_of_lines,2))
    label_vector = []
    i = 0
    for row in array_of_lines:
        row = row.strip('\n')
        row = row.split(',')
        ret_matrix[i,:] = row[0:2]
        label_vector.append(row[-1])
        i += 1
    return ret_matrix, label_vector

def take_first(ele):
    return ele[0]

def kNN_classify(k, sample_matrix, sample_labels, data_matrix):
    result_labels = [None] * len(data_matrix)

    for i in range(len(data_matrix)):
        dist = [0] * len(sample_matrix)  # element: tuple (distance, label)
        # Calculate data_matrix[i]'s distance to each sample point
        for j in range(len(sample_matrix)):
            for k in range(len(sample_matrix[0])):
                dist[j] += (data_matrix[i,k] - sample_matrix[j,k])**2
            dist[j] = sqrt(dist[j]), sample_labels[j]
        
        # Sort
        dist.sort(key = take_first)

        # Count the first k items
        cnt_a = 0
        cnt_b = 0
        for t in dist[0:k]:
            if t[1] == 'A':
                cnt_a += 1
            elif t[1] == 'B':
                cnt_b += 1
        if cnt_a > cnt_b:
            result_labels[i] = 'A'
        elif cnt_a < cnt_b:
            result_labels[i] = 'B'
        else:
            result_labels[i] = None

    return result_labels

if __name__ == '__main__':
    
    # Move to the directory of kNN.py
    path = os.path.abspath(sys.argv[0])
    dir = os.path.split(path)[0]
    os.chdir(dir)

    # Load data from csvfile
    data_matrix, data_labels = csv2matrix('./dataset_knn.csv')
    
    # Sampling (10% as sample data set)
    sample_matrix, sample_labels = sampling.huang_sampling(data_matrix, data_labels, len(data_matrix) // 10)
    
    # kNN
    result_labels = kNN_classify(10, sample_matrix, sample_labels, data_matrix)

    # Evaluate the result
    # Using mat = numpy.row_stack((mat, row)) to add rows to matrix.
    true_positive = empty((0,len(data_matrix[0])))
    true_negative = empty((0,len(data_matrix[0])))
    false_positive = empty((0,len(data_matrix[0])))
    false_negative = empty((0,len(data_matrix[0])))
    failed_to_classify = empty((0,len(data_matrix[0])))

    for i, label in enumerate(result_labels):
        if label == 'A' and label == data_labels[i]:
            true_positive = row_stack((true_positive, data_matrix[i]))
        elif label == 'B' and label == data_labels[i]:
            true_negative = row_stack((true_negative, data_matrix[i]))
        elif label == 'A' and label != data_labels[i]:
            false_positive = row_stack((false_positive, data_matrix[i]))
        elif label == 'B' and label != data_labels[i]:
            false_negative = row_stack((false_negative, data_matrix[i]))
        else:
            # Failed to classify (label == None)
            failed_to_classify.append(data_matrix[i])
    num_TP = len(true_positive)
    num_TN = len(true_negative)
    num_FP = len(false_positive)
    num_FN = len(false_negative)
    print('Sensitivity: ' + str(num_TP / (num_TP + num_FN)))
    print('Specificity: ' + str(num_TN / (num_TN+num_FP)))
    print('Precision  : ' + str(num_TP / (num_TP+num_FP)))
    print('Accuracy   : ' + str((num_TP + num_TN) / (num_TP + num_TN + num_FP + num_FN)))
    
    # Show figures
    fig = plt.figure()

    ground_truth_color = [None] * len(data_labels)
    for i, label in enumerate(data_labels):
        if label == 'A':
            ground_truth_color[i] = 'blue'
        elif label == 'B':
            ground_truth_color[i] = 'yellow'
    ground_truth_plot = fig.add_subplot(131)
    ground_truth_plot.set_title('Ground Truth')
    ground_truth_plot.scatter(data_matrix[:, 0], data_matrix[:, 1], s=5, c=ground_truth_color)

    sample_color = [None] * len(sample_labels)
    for i, label in enumerate(sample_labels):
        if label == 'A':
            sample_color[i] = 'blue'
        elif label == 'B':
            sample_color[i] = 'yellow'
    sapmle_plot = fig.add_subplot(132)
    sapmle_plot.set_title('Sampling')
    sapmle_plot.scatter(data_matrix[:, 0], data_matrix[:, 1], s=5, c='gray')
    sapmle_plot.scatter(sample_matrix[:, 0], sample_matrix[:, 1], s=5, c=sample_color)

    result_plot = fig.add_subplot(133)
    result_plot.set_title('k-NN Result')
    result_plot.scatter(true_positive[:, 0], true_positive[:, 1], s=5, c='blue', label='True Positive')
    result_plot.scatter(true_negative[:, 0], true_negative[:, 1], s=5, c='yellow', label='True Negative')
    result_plot.scatter(false_positive[:, 0], false_positive[:, 1], s=5, c='red', marker='^', label='False Positive')
    result_plot.scatter(false_negative[:, 0], false_negative[:, 1], s=5, c='purple', marker='^', label='False Negative')

    plt.legend(loc='best')  # set the position of the legend
    plt.show()