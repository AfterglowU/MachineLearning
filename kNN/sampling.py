from numpy import *
import random

def random_sampling(matrix, labels, num):

    sample_matrix = zeros((num, len(matrix[0])))
    sample_labels = [None] * num
    selected = [0] * len(matrix)
    
    i = 0
    while i < num:
        # Select a unselected vector from the matrix
        rand = random.randint(0,len(matrix)-1)
        while selected[rand]:
            rand = random.randint(0,len(matrix)-1)
        selected[rand] = 1
        sample_matrix[i,:] = matrix[rand,:]
        sample_labels[i] = labels[rand]
        i += 1
    
    return sample_matrix, sample_labels

def huang_sampling(matrix, labels, num):
    
    '''
    An optimized sampling method introduced by Mr.Huang.
    This method will sample more uniformly.
    Time complexity: O(kN), k is k-NN's k and N is size of the dataset.
    '''

    sample_matrix = zeros((num, len(matrix[0])))
    sample_labels = [None] * num
    selected = [0] * len(matrix)
    
    # Select the first point randomly
    rand = random.randint(0, len(matrix)-1)
    selected[rand] = 1
    sample_matrix[0,:] = matrix[rand,:]
    sample_labels[0] = labels[rand]

    # Select the rest num-1 points.
    # Never use the annotated code below, it's too slow.
    '''
    k = 1
    while k < num:
        mindist = [float('inf')] * len(matrix)
        # Calculate the min dist to "k" selected points of each unselected one.
        for i in range(len(matrix)):
            if not selected[i]:
                for j in range(k):
                    dist = 0
                    for n in range(len(matrix[0])):
                        dist += (matrix[i,n] - sample_matrix[j,n])**2
                    dist = sqrt(dist)
                    if dist < mindist[i]:
                        mindist[i] = dist
    '''

    # Select the rest num-1 points.
    # A faster version. You don't need to calculate dist to sample[0] ~ sample[k-1] in each loop, 
    # since the mindist to the prior k-1 selected points (aka. sample[0] ~ sample[k-1]) has already been calculated in previous loops.
    # Just calculate dist to sample[k] and compare it with the mindist[i].
    mindist = [float('inf')] * len(matrix)
    k = 1
    while k < num:
        # Calculate the min dist to "k" selected points of each unselected one.
        for i in range(len(matrix)):
            if not selected[i]:
                dist = 0
                for n in range(len(matrix[0])):
                    dist += (matrix[i,n] - sample_matrix[k-1,n])**2
                dist = sqrt(dist)
                if dist < mindist[i]:
                    mindist[i] = dist
        # Choose the point with "the farthest mindist"
        for i in range(len(matrix)):
            if not selected[i]:
                choice = i
                break
        for i in range(choice + 1, len(matrix)):
            if not selected[i] and mindist[i] > mindist[choice]:
                choice = i
        selected[choice] = 1
        sample_matrix[k,:] = matrix[choice,:]
        sample_labels[k] = labels[choice]
        k += 1

    return sample_matrix, sample_labels
