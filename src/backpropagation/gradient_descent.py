from models.n_classifier.classifier import lin_model, layer
import numpy as np
import math
from collections import deque
# input shape: list[(input vector: list[float], actualOutput: list[float])], actualOutput is the numerical rep of intents
# if actual is only reminder intent: [1.,0,0]
# if intent is both reminder and email: [1.,1.,0]
# if not reminder and email: [0,0,1.]
def mini_batch_gradient_descent(model: lin_model, training_inputs: np.ndarray, test_samples, test_function, batch_size=5):
    total_size = len(training_inputs)
    remainder = total_size % batch_size
    if remainder != 0:
        print('WARNING: some batch_sizes are not fully filled')
        
    # 1 batch per epoch; for every epoch
    for i in range(math.ceil(total_size/batch_size)):
        start = math.ceil(i*batch_size)
        # iterate inputs within batch
        print(len(model.outputs), len(model.layers))
        model.feed_forward(training_inputs[start][0])
        h_1st_lyr_grads, grads, output_lyr_grads = model.get_gradients(training_inputs[start][1])
        model.clear_cache()
        for entry in training_inputs[start+1:min(start+batch_size,total_size)]:
            model.feed_forward(entry[0])
            h_1_l_g, grad, op_l_g = model.get_gradients(entry[1])    
            h_1st_lyr_grads += h_1_l_g
            grads += grad
            output_lyr_grads += op_l_g        
            model.clear_cache()
        avg_grads = [np.divide(h_1st_lyr_grads,batch_size), np.divide(grads,batch_size), np.divide(output_lyr_grads,batch_size)]
        #start updating weights layer by layer        
        update_weights(model, avg_grads)
        print(f'EPOCH #{i+1} weights:',model.get_weights())
        test_function(model, test_samples, f'EPOCH #{i+1}:')


def update_weights(model: lin_model, grads_tuple):
    assert model.total_layers > 0
    start_lyr: layer = model.get_layer(0)
    start_lyr.weights -= model.learning_rate*grads_tuple[0]
    
    for i in range(1, model.total_layers-1):
        lyr: layer = model.get_layer(i)
        lyr.weights -= model.learning_rate*grads_tuple[1][i-1]

    op_lyr: layer = model.get_layer(-1)
    op_lyr.weights -= model.learning_rate*grads_tuple[-1]
        