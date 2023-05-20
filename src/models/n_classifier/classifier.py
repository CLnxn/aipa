import nltk
import json
import numpy as np
from collections import deque
import sys
sys.path.insert(0, 'C:\\Users\\proki\\repos\\aipa\\src')
from optimisers.activation_fn import sigmoid
class lin_model():
    def __init__(self, learning_rate=0.01,
                 input_size=200, layer_size = 75
                 ) -> None:
        self.total_layers = 0
        self.layer_size = layer_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.layers = deque()
    # adds a layer (can be hidden | output layer)
    def add_layer(self, activation_fn, layer_size: int =None):
        if layer_size == None:
            layer_size = self.layer_size

        new_layer = layer(activation_fn,layer_size)
        self.total_layers+=1
        #if first layer is being added, init weights shape based on input size
        if not self.layers:
            new_layer.initWeights(self.input_size)
            self.layers.append(new_layer)
            return
        prev_layer: layer = self.layers[-1]
        new_layer.initWeights(prev_layer.total_nodes) #prev is assumed to be dense
        self.layers.append(new_layer)
    #get index by layer
    def get_layer(self, layer_index: int):
        return self.layers[layer_index] 
    #gets model output from a numerically embedded input
    def feed_forward(self, input: list[float] , lyr=0, activation_fn=sigmoid):
        #return input after going through last layer
        print(input)
        if lyr > self.total_layers:
            return input
        #input layer processing
        if lyr == 0:
            n = self.input_size - len(input)
            if n < 0:
                return -1
            if n > 0:
                pad_input(input, n)
            # print(len(input),self.input_size)

            outputs = deque()
            for i, entry in enumerate(input):
                out = activation_fn(entry)
                outputs.append(out)
            return self.feed_forward(outputs, lyr+1) 
        #hidden layer processing
        curr_layer: layer = self.get_layer(lyr-1)
        return self.feed_forward(curr_layer.computeOutputs(input),lyr+1)
    
    def pad_input(input: list, amount: int, fill = 0):
        for i in range(amount):
            input.append(fill)        


class layer():
    def __init__(self, activation_fn, layer_size):
        self.weights = []
        self.bias = 1
        self.activation_fn = activation_fn
        self.total_nodes = layer_size
    # initialises weights
    def initWeights(self, size:int, value=1.0):
        self.weights = np.full([self.total_nodes,size],value)
    #computes outputs for every node for the layer using weights. weights has to be initalised before calling.
    def computeOutputs(self, input: deque):
        outputs = deque()
        if len(self.weights) <= 0:
            print('b')
            return -1
        n = self.weights.shape[1] - len(input)
        print(self.weights.shape,len(input))
        if n < 0:
            print('a')
            return -1
        if n > 0:
            pad_input(input, n)
        for node_i in range(self.total_nodes):
            a = np.dot(self.weights[node_i], input)+ self.bias
            outputs.append(self.activation_fn(a))
        return outputs
#pads a <fill> value <amount> times behind the input
def pad_input(input: list , amount: int, fill = 0):
        for i in range(amount):
            input.append(fill)  
def lin_NN_model(input_sentence:str):
    #lemmatise the input
    intents =  get_intents()

    tokens = nltk.word_tokenize(input_sentence)    
    return 0


def get_intents():
    intents_fv = open('../intents/intents.json', "r")
    intents_input = intents_fv.read()
    intents_fv.close()
    return json.loads(intents_input)



