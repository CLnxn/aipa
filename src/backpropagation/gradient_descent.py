from models.n_classifier.classifier import lin_model
import numpy as np
import math
def gradient_descent(model: lin_model, input: np.ndarray, epochs=5):
    total_size = len(input)
    if total_size % epochs != 0:
        print('WARNING: actual epoch shape is not fully 5')
    for i in range(math.ceil(total_size/epochs)):
        start = np.divide(i,epochs)*total_size
        for j, entry in enumerate(input[start:start+epochs]):
            pass 

