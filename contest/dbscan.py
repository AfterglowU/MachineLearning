import numpy as np
import random

def get_neighbour(data_array:np.array, center:np.array, eps:float) -> list:
    ret = set()
    for i in range(data_array.shape[0]):
        if np.linalg.norm(center-data_array[i]) < eps:
            ret.add(i)
    return ret

def dbscan(data_array:np.array, eps:float, minpoints:int) -> np.array:
    core_objects = {}   # key  : index of this core object in data_array
                        # value: A set of indices of objects in it's epsilon-neighbourhood

    C = {}              # key  : class label
                        # value: A set of indices of objects that belong to this class

    n = data_array.shape[0]

    # Find all core objects
    for i in range(n):
        neighbour = get_neighbour(data_array, data_array[i], eps)
        if len(neighbour) >= minpoints:
            core_objects[i] = neighbour
    
    k = 0
    not_visited = {i for i in range(n)}
    while core_objects:
        not_visited_old = not_visited.copy()
        # Pick one core object randomly
        cores = list(core_objects.keys())
        rand = random.randint(0,len(cores)-1)
        core = cores[rand]
        queue = [core]
        not_visited.remove(core)
        while queue:
            q = queue[0]
            del queue[0]
            if q in core_objects:
                delta = core_objects[q] & not_visited
                queue.extend([i for i in delta])
                not_visited -= delta
        # Generate cluster k
        k += 1
        C[k] = not_visited_old - not_visited
        for i in C[k]:
            if i in core_objects.keys():
                del core_objects[i]
        
    # Return a list of labels rather than clustering partition C
    labels = [-1 for i in range(n)]
    for i in range(1,k+1):
        for j in C[i]:
            labels[j] = i

    return labels