import numpy as np
from optimisers.activation_fn import sigmoid_derivative
from collections import deque

def output_sqred_err_grad_vec(act_fn_d, 
                                  calc_y_vect: list[float], 
                                  act_y_vect: list[float],
                                  activ_lyr_vect:list[float], 
                                  prev_lyr_op_vect: list[float]):
    """used to compute the avg gradients & avg deltas of the weights of an n-node output layer 
    in the output layer only.

    **All array orderings follow top to down (corresponds index 0 to len-1) 
    for a feedforward network**
    
    Args:
        act_fn_d (function): derivative of activation fn used in network
        calc_y_vect: vector of calculated y values (float), dimension equates to no. of nodes in 
                     output layer
        act_y_vect: vector of actual y values (float) obtained from training data,
                     same dimensions as calc_y_vect,
        activ_lyr_vect: vector containing the output layer's activation values for each node
        prev_lyr_op_vect: vector containing the previous layer's output values from each node 
                          (assumes dense layer, and excludes additional last term for bias)
    Returns:
        2-element tuple of (deltas for each node in the output layer,
        2d array: an array of nodes containing array of errors wrt weights in that node)
    """
    print('start of output_sqred_arr_vec')
    print(calc_y_vect,act_y_vect,activ_lyr_vect, prev_lyr_op_vect)
    prev_lyr_op_vect = deque(prev_lyr_op_vect)
    prev_lyr_op_vect.append(1)
    act_d_op = np.asarray([act_fn_d(activation) for activation in activ_lyr_vect])
    gradients = np.zeros((len(activ_lyr_vect), len(prev_lyr_op_vect)))
    d_y = np.subtract(calc_y_vect,act_y_vect)
    deltas = deque()
    # for each node in op layer (no. of nodes shd also equal len of calc_y_vec)
    for i in range(len(activ_lyr_vect)):
        mean_err = np.mean(d_y)
        # compute mean deltaNode for each node in the layer
        deltaNode = mean_err*act_d_op[i]
        deltas.append(deltaNode)
        gradients[i] = np.multiply(deltaNode,prev_lyr_op_vect)
        
        # move on to the next node
    print('end of output_sqred_arr_vec')
    
    return deltas, gradients

def sqred_err_grad_vec(act_fn_d, 
                       curr_lyr_activ_vect, 
                       next_lyr_w_matrix:np.ndarray, 
                       next_lyr_deltas, prev_lyr_op_vect):
    """used to compute the avg gradients & avg deltas of the weights of nodes in a
    particular hidden layer.


    **All array orderings follow top to down (corresponds index 0 to len-1) 
    for a feedforward network**
    Args:
        act_fn_d: derivative function (fn) of the current layer's activation fn
        curr_lyr_activ_vect: vector of activations of nodes in the current layer
        next_lyr_w_matrix: 2d array of weights; an array of array of weights from every node of 
                            the next layer connected to a particular node in the curr layer 
                            (transpose of a regular matrix of the next layer nodes's weights)
        next_lyr_deltas: array of deltas for each node in the next layer
        prev_lyr_op_vect: vector containing output of nodes from prev layer (assumes dense layer, and excludes additional last term for bias)

    Returns:
        2-element tuple of (deltas for each node in the current layer,
        2d array: an array of nodes containing array of errors wrt weights in that node)
    """
    print('before test',next_lyr_w_matrix.shape, next_lyr_deltas)
    
    # start of tests
    # 1. next layer same shape
    if int(next_lyr_w_matrix.shape[1]) != len(next_lyr_deltas):
        raise Exception('Failed test 1: matrix x-axis (length of array element) shape not equal next layer size.')
    # 2. curr layer same shape
    if int(next_lyr_w_matrix.shape[0]) < len(curr_lyr_activ_vect):
        raise Exception(f'Failed test 2: matrix y-axis (outer array length) shape not equal to curr layer size. {int(next_lyr_w_matrix.shape[0])} vs {len(curr_lyr_activ_vect)+1}')
    # end of tests

    #adding op term=1 for bias at the end during dot product
    prev_lyr_op_vect = deque(prev_lyr_op_vect)
    prev_lyr_op_vect.append(1)

    gradients = np.zeros((len(curr_lyr_activ_vect), len(prev_lyr_op_vect)))
    deltas = deque()
    # start populating gradients and deltas using backprog to from next (right) to current layer
    # print('start 0')
    for i, activation in enumerate(curr_lyr_activ_vect):
        # print('start 1')
        activ_d = act_fn_d(activation)
        # sum of next layer weights associated with curr layer node with next layer deltas 
        next_lyr_sum = np.dot(next_lyr_w_matrix[i], next_lyr_deltas)
        # check scalar
        if not type(next_lyr_sum) is np.float64:
            raise Exception('dot product does not produce a scalar value') 
        
        deltaNode = next_lyr_sum*activ_d
        deltas.append(deltaNode)
        # print(next_lyr_w_matrix[i], next_lyr_deltas)
        gradients[i] = np.multiply(deltaNode, prev_lyr_op_vect) 
        # print('end 1')
    # print('end 0')
    return deltas, gradients

# val = output_sqred_err_grad_vec(sigmoid_derivative, 
#                                [0.8807971, 0.9644288, 0.982014], 
#                                [1,1,0.5], [2.0,3.3,4], [0.7,0.8,0.2])

# test2 = sqred_err_grad_vec(sigmoid_derivative, [5.0,4.0,1.4,2.2],np.full((4,3),5.0),
#                            [0.8807971, 0.9644288, 0.982014],[0.8,0.3,0.4,0.8])
# print(test2)
# print(val)