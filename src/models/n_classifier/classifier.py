import nltk
import json
import numpy as np
from collections import deque
import sys
sys.path.insert(0, 'C:\\Users\\proki\\repos\\aipa\\src')
from optimisers.activation_fn import sigmoid, sigmoid_derivative
from optimisers.loss import output_sqred_err_grad_vec, sqred_err_grad_vec

class lin_model():
    def __init__(self, learning_rate=0.01,
                 input_size=200, layer_size = 75
                 ) -> None:
        self.total_layers = 0
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = deque()
        self.outputs = []
        self.activations = deque()
    # adds a layer (can be hidden | output layer)
    def add_layer(self, activation_fn, activation_fn_d,layer_size: int =None):
        if layer_size == None:
            layer_size = self.layer_size

        new_layer = layer(activation_fn, activation_fn_d, layer_size)
        self.total_layers+=1
        # if first layer is being added, init weights shape based on input size
        if not self.layers:
            new_layer.initWeights(self.input_size)
            self.layers.append(new_layer)
            return
        prev_layer: layer = self.layers[-1]
        new_layer.initWeights(prev_layer.total_nodes) #prev is assumed to be dense
        self.layers.append(new_layer)
    
    # get index by layer
    def get_layer(self, layer_index: int):
        return self.layers[layer_index] 
    # gets model output from a numerically embedded input
    def feed_forward(self, input: list[float], lyr=0, activation_fn=sigmoid):
        # return input after going through last layer
        print(input)
        if lyr > self.total_layers:

            return input
        # input layer processing
        if lyr == 0:
            n = self.input_size - len(input)
            if n < 0:
                return -1
            if n > 0:
                pad_input(input, n)
            # print(len(input),self.input_size)

            outputs = deque()
            for i, entry in enumerate(input):
                activation = activation_fn(entry)
                outputs.append(activation)
            self.activations.append(outputs) # store activation of input layer in index 0
            self.outputs.append(outputs) # store output of input layer in index 0
            return self.feed_forward(outputs, lyr+1) 
        
        # hidden layer index is lyr-1, lyr >= 1
        curr_layer: layer = self.get_layer(lyr-1)

        outputs = curr_layer.computeOutputs(input)
        # store output of completed layer i in index i
        self.outputs.append(outputs) 
        # store activations of completed layer i in index i
        self.activations.append(curr_layer.activations)
        # clear cached activations in layer
        curr_layer.clear_activations_store()
        return self.feed_forward(outputs,lyr+1)
    
    def pad_input(input: list, amount: int, fill = 0):
        for i in range(amount):
            input.append(fill)        
    def clear_outputs(self):
        self.outputs.clear()
        self.outputs = []
    def clear_activations(self):
        self.activations.clear()
        self.activations = deque()

    def clear_cache(self):
        self.clear_activations()
        self.clear_outputs()

    # returns a gradients array for every weight in every node in every layer (3d array)
    def get_gradients(self, actual_y_vect:list[float]):
        gradients = deque()
        deltas = deque()
        count = len(self.outputs)-1
        for i in range(len(self.layers)-1,-1,-1):
            #if output layer
            if count == len(self.outputs)-1:
                dels, grads = output_sqred_err_grad_vec(sigmoid_derivative, self.outputs[i],
                                          actual_y_vect, self.layers[i-1])
                gradients.appendleft(grads)
                deltas.appendleft(dels)
                count-= 1
                continue
            next_layer: layer = self.get_layer(i+1)
            transposed_weights = np.transpose(next_layer.weights)            
            dels, grads = sqred_err_grad_vec(sigmoid_derivative, 
                                             self.outputs[i],
                                             transposed_weights,
                                             deltas[i+1],
                                             self.outputs[count-1])
            count -= 1
            deltas.appendleft(dels)
            gradients.appendleft(grads)
        return gradients

class layer():
    def __init__(self, activation_fn, activation_fn_d, layer_size):
        self.weights = []
        self.bias = 1
        self.activation_fn = activation_fn
        self.activation_fn_d = activation_fn_d 
        self.total_nodes = layer_size
        self.activations = deque()
    # initialises weights such that there are <size> weights per node in the layer
    def initWeights(self, size:int, value=1.0):
        self.weights = np.full([self.total_nodes,size],value)
    # computes outputs for every node for the layer using weights. weights has to be initalised before calling.
    # self.activations is populated everytime computeOutputs is called
    def computeOutputs(self, input: deque):
        outputs = deque()
        # reject if uninitialised
        if len(self.weights) <= 0:
            print('b')
            return -1
        n = self.weights.shape[1] - len(input)
        print(self.weights.shape,len(input))
        # reject if not enough weights to calculate activation 
        if n < 0:
            print('a')
            return -1
        # pad default fill value (0) to the end of input if input size < weights size 
        if n > 0:
            pad_input(input, n)
        # compute activation for each node 
        # plug into activation fn to get output
        # appends them into an outputs vector
        for node_i in range(self.total_nodes):
            a = np.dot(self.weights[node_i], input)+ self.bias

            # stores activation into class
            self.activations.append(a)

            outputs.append(self.activation_fn(a))

        return outputs
    # clears activations field  when called. 
    def clear_activations_store(self):
        self.activations.clear()
        self.activations = deque()


# pads a <fill> value <amount> times behind the input
def pad_input(input: list , amount: int, fill = 0):
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



