from models.n_classifier.classifier import lin_model
import numpy as np
import math
# input shape: list[(list[float], actualOutput: list[float])], actualOutput is the numerical rep of intents
# if actual is only reminder intent: [1.,0,0]
# if intent is both reminder and email: [1.,1.,0]
# if not reminder and email: [0,0,1.]
def mini_batch_gradient_descent(model: lin_model, input: np.ndarray, batch_size=5):
    total_size = len(input)
    remainder = total_size % batch_size
    if remainder != 0:
        print('WARNING: some batch_sizes are not fully filled')
        

    for i in range(math.ceil(total_size/batch_size)):
        start = math.ceil(i*batch_size)
        # iterate inputs within batch
        outs = [], gradients = []
        for j, entry in enumerate(input[start:min(start+batch_size,total_size)]):
            calculated_y = model.feed_forward(entry[0])
            outs.append(model.outputs[:-1]) # exclude outputs from the output layer
            
            model.clear_outputs()

            pass   



        