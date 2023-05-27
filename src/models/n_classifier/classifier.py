import nltk
import json
import numpy as np
from collections import deque
import sys
sys.path.insert(0, 'C:\\Users\\proki\\repos\\aipa\\src')
sys.path.insert(0, 'D:\\proki\\repos\\general\\aipa\\src')
from optimisers.activation_fn import sigmoid, sigmoid_derivative
from optimisers.loss import output_sqred_err_grad_vec, sqred_err_grad_vec

class lin_model():
    def __init__(self, learning_rate=0.01,
                 input_size=200, layer_size = 75
                 ) -> None:
        """ Instantiates a linear classification model with dense layers with default hyper params.

        Args:
            total_layers: current layers present in the network.
            layer_size: default size when adding a new layer to the network.
            learning_rate: default learning rate used in the model (for updating weights).
            input_size: default number of nodes in the input layer (as an array with elements as nodes holding a numerical value (float)).
            layers: deque collection of <layer> class layers in the model (including hidden and output layer, input layer not counted).
            outputs: deque collection (collection of layers) of deque collections (output of nodes within each layer in the network) of outputs.
            activation: deque collection (collection of layers) of deque collections (activation of nodes within each layer in the network)
            of activations.
            
        
        """
        self.total_layers = 0
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = deque()
        self.outputs = deque() # same length as activations
        self.activations = deque() # same length as outputs
    # adds a layer (can be hidden | output layer)
    def add_layer(self, activation_fn, activation_fn_d,layer_size: int =None):
        """Inserts a dense layer (hidden or output layer determined by order of insertion by calling this method) 
        into the network from the end.

        Args:
            activation_fn: the activation function used to compute the output of each node in the layer.
            activation_fn_d: the activation function's derivative w.r.t activation.
            layer_size: the number of nodes in the layer. (Defaults to self.layer_size).
        """
        if layer_size == None:
            layer_size = self.layer_size

        new_layer = layer(activation_fn, activation_fn_d, layer_size)
        self.total_layers+=1
        # if first layer is being added, init weights shape based on input size
        if not self.layers:
            new_layer.initWeights(self.input_size+1)
            self.layers.append(new_layer)
            return
        prev_layer: layer = self.layers[-1]
        new_layer.initWeights(prev_layer.total_nodes+1) #prev is assumed to be dense + bias
        self.layers.append(new_layer)
    
    # get index by layer
    
    def get_layer(self, layer_index: int):
        """
        get layer by index order (zero-indexed, with index 0 being the first hidden layer, and length-1 being the output layer).
        
        Args:
            layer_index: index of the layer instance to retrieve
        Returns:
            (layer): the indexed layer class instance
        """
        return self.layers[layer_index]
    # gets model output from a numerically embedded input
    def feed_forward(self, input: list[float], lyr=0, activation_fn=sigmoid, cleanAfter=False):
        """gets a model output vector (calculated y) by feeding in a numerically embedded input (input) provided in the args through the
          network's layers. This method populates the self.outputs & self.activations property when called. 
          These need to be manually cleared by calling clear_cache() after processing and gradient descent for 1 data input is completed
          to prevent any inaccurate overwriting of data in subsequent feed_forward and any loss optimisation method calls.
        Args: 
            input: the input vector to be forward-processed in the network
            lyr: the internal index to track the current layer of the function's iteration
            activation_fn: the activation function to be used in the input layer
            cleanAfter (bool): determines whether the model will clear the activations & outputs stored after a feedforward iteration
        Returns:
            an output vector from the network
        """
        # return input after going through last layer
        # print(input)
        if lyr > self.total_layers:
            if cleanAfter:
                self.clear_cache()
            return input
        # input layer processing
        if lyr == 0:
            n = self.input_size - len(input)
            if n < 0:
                # if the input is larger than the available nodes, consider only the first self.input_size entries
                input = input[:n]
            if n > 0:
                pad_input(input, n)
            # print(len(input),self.input_size)

            outputs = deque()
            for i, entry in enumerate(input):
                activation = activation_fn(entry)
                outputs.append(activation)
            self.activations.append(input) # store activation of input layer in index 0
            self.outputs.append(outputs) # store output of input layer in index 0
            return self.feed_forward(outputs, lyr+1, cleanAfter=cleanAfter) 
        
        # hidden layer index is lyr-1, lyr >= 1
        curr_layer: layer = self.get_layer(lyr-1)

        outputs = curr_layer.computeOutputs(input)
        # store output of completed layer i in index i
        self.outputs.append(outputs) 
        # store activations of completed layer i in index i
        self.activations.append(deque(curr_layer.activations))
        # clear cached activations in layer
        curr_layer.clear_activations_store()
        return self.feed_forward(outputs,lyr+1, cleanAfter=cleanAfter)
    
    def pad_input(input: list, amount: int, fill = 0):
        for i in range(amount):
            input.append(fill)        
    def clear_outputs(self):
        self.outputs.clear()
    def clear_activations(self):
        self.activations.clear()

    def clear_cache(self):
        self.clear_activations()
        self.clear_outputs()

    def get_gradients(self, actual_y_vect:list[float], clean=True):
        """Called after running feed_forward to obtain the gradients of every node in the layer for further optimisation of weights.
        This method can ONLY be called after running feed_forward, as it uses internal <activations> & <outputs> variables 
        for computing the various gradient changes per node.
        
        Args: 
            actual_y_vect: list of actual output values to be used with the network-computed output values to 
            obtain the error and gradients for updating the weights.
            clean: determines whether to clean the inhomogeneous gradients array
            into a homogeneous np array tuple (1st hidden_layer, hidden_layers,output_layer) .
        Returns: 
            A gradients array for every weight in every node in every layer (3d array).

            Each ndarray element's 3D shape follows (number of layers, number of nodes per layer, number of weights per node per layer)\n
            Each ndarray element's 2D shape follows (number of nodes in layer, number of weights per node in layer)
        
        """
        gradients = deque()
        deltas = deque() # higher index = closer to output layer, len(deltas)-1 = output layer delta
        assert len(self.outputs)-1 == len(self.activations)-1
        print(f'{len(self.outputs)} {len(self.layers)+1}')
        assert len(self.outputs) == len(self.layers)+1
        count = len(self.outputs)-1
        #print('output after feedforward', self.outputs)
        for i in range(len(self.layers)-1,-1,-1):
            #if output layer
            if count == len(self.outputs)-1:
                dels, grads = output_sqred_err_grad_vec(sigmoid_derivative, self.outputs[count],
                                          actual_y_vect, self.activations[count],self.outputs[count-1])
                #print('dels, grads:',dels,grads)
                gradients.appendleft(grads)
                deltas.appendleft(dels)
                count-= 1
                continue
            next_layer: layer = self.get_layer(i+1)
            transposed_weights = np.transpose(next_layer.weights)            
            dels, grads = sqred_err_grad_vec(sigmoid_derivative, 
                                             self.activations[count],
                                             transposed_weights,
                                             deltas[0],
                                             self.outputs[count-1])
            count -= 1
            deltas.appendleft(dels)
            gradients.appendleft(grads)
        # weights per node depends on input layer node size, thus different from the other hidden layers. Removed (popleft) for numpy array homogenity
        hidden_layer_1 = np.asarray(gradients.popleft()) 
        # layer size differs from the other
        output_grad = np.asarray(gradients.pop())
        grads = np.asarray(gradients)
        print('Gradient calculations completed \n\n')
        print('1st layer gradients:',hidden_layer_1, '\n\n')
        print('hidden layer gradients:', grads, '\n\n')
        print('output layer gradients:', output_grad, '\n\n')
        print('gradient shapes:',f'{hidden_layer_1.shape}, {grads.shape}, {output_grad.shape}')
        
        return hidden_layer_1, grads, output_grad
    
    def get_gradients_v2(self, actual_y_vect:list[float], outputs: deque, activations: deque):
        """Called after running feed_forward to obtain the gradients of every node in the layer for further optimisation of weights.
        This method can ONLY be called after running feed_forward, as it uses internal <activations> & <outputs> variables 
        for computing the various gradient changes per node.
        
        Args: 
            actual_y_vect: list of actual output values to be used with the network-computed output values to 
            obtain the error and gradients for updating the weights.
        Returns: 
            a gradients array for every weight in every node in every layer (3d array)
        
        
        """
        gradients = deque()
        deltas = deque() # higher index = closer to output layer, len(deltas)-1 = output layer delta
        assert len(outputs)-1 == len(activations)-1
        assert len(outputs) == len(self.layers)+1
        count = len(outputs)-1
        for i in range(len(self.layers)-1,-1,-1):
            #if output layer
            if count == len(outputs)-1:
                dels, grads = output_sqred_err_grad_vec(sigmoid_derivative, outputs[count],
                                        actual_y_vect, activations[count], outputs[count-1])
                print('dels, grads:',dels,grads)
                gradients.appendleft(grads)
                deltas.appendleft(dels)
                count-= 1
                continue
            next_layer: layer = self.get_layer(i+1)
            transposed_weights = np.transpose(next_layer.weights)            
            dels, grads = sqred_err_grad_vec(sigmoid_derivative, 
                                             activations[count],
                                             transposed_weights,
                                             deltas[0],
                                             outputs[count-1])
            count -= 1
            deltas.appendleft(dels)
            gradients.appendleft(grads)
        return gradients

    def get_weights(self):
        """
        Returns:
            3D array of arrays (layers) that each (node) contain arrays of weights of a node.
        """
        weights = deque()
        for i, lyr in enumerate(self.layers):
            lyr: layer
            weights.append(lyr.weights)
        return weights
class layer():
    def __init__(self, activation_fn, activation_fn_d, layer_size):
        self.weights = []
        self.activation_fn = activation_fn
        self.activation_fn_d = activation_fn_d 
        self.total_nodes = layer_size
        self.activations = deque()

    def initWeights(self, size:int, value=0.001):
        """Initialises weights such that there are <size> weights per node in the layer
        
        Args: 
            size: the number of weights in each node of the layer (excludes bias). (Usually equal to the number of nodes in the previous layer.)
            value: the default value for each weight. 
        """

        self.weights = np.full([self.total_nodes,size],value)

    def computeOutputs(self, input: deque):
        """Computes outputs for every node for the layer using weights. 
        
        Note: 
            1. Weights has to be initalised before calling.
            2. self.activations is populated everytime computeOutputs is called.
        
        Args: 
            input: queue collection of outputs from the previous layer to be used in calculating the output for the current layer.

        Returns: a deque collection of outputs from the current layer.
        """
        outputs = deque()
        
        

        # reject if uninitialised
        if len(self.weights) <= 0:
            print('Error in computeOutputs: weights are not initialised')
            return -1
        n = self.weights.shape[1] - len(input) - 1
        # print(self.weights.shape,len(input))
        # reject if not enough weights to calculate activation 
        if n < 0:
            print(f'Error in computeOutputs: not enough weights to calculate activation. n: {n}')
            return -1
        # pad default fill value (0) to the end of input if input size < weights size 
        if n > 0:
            pad_input(input, n)
        # append additional input el for bias
        input.append(1)
        # compute activation for each node 
        # plug into activation fn to get output
        # appends them into an outputs vector
        for node_i in range(self.total_nodes):
            a = np.dot(self.weights[node_i], input)

            # stores activation into class
            self.activations.append(a)

            outputs.append(self.activation_fn(a))
        # remove the op term for bias from queue reference
        input.pop()
        return outputs
     
    def clear_activations_store(self):
        """Clears activations field  when called."""
        self.activations.clear()

def pad_input(input: list , amount: int, fill = 0):
        """Pads a <fill> value <amount> times behind the input
        
        Args:
            input: list input to pad values to
            amount: the amount of times to pad the list 
            fill: the value used to pad the list
        """
        for i in range(amount):
            input.append(fill)  
def lin_NN_model(input_sentence:str):
    # lemmatise the input
    intents =  get_intents()

    tokens = nltk.word_tokenize(input_sentence)    
    return 0


def get_intents():
    intents_fv = open('../intents/intents.json', "r")
    intents_input = intents_fv.read()
    intents_fv.close()
    return json.loads(intents_input)



