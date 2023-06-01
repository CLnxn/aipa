import numpy as np
from numpy.random import rand
def xavier_init(prev_n, curr_size, bias_val = 0):
    u = np.divide(1,np.sqrt(prev_n))
    l = -u
    weights = rand(curr_size, prev_n+1)
    scaled = l + weights*(2*u)

    flat_w = np.ndarray.flatten(scaled)
    flat_w.mean()
    print(flat_w.mean(), flat_w.std(), flat_w.var())
    for node_w in scaled:
        node_w[-1] = bias_val    
    return scaled

# print(xavier_init(5,10))