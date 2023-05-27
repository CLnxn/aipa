import pandas as pd
import json
from classifier import layer, lin_model
import random
import sys
import numpy as np
print(sys.path)
sys.path.insert(0, 'C:\\Users\\proki\\repos\\aipa\\src')
sys.path.insert(0, 'D:\\proki\\repos\\general\\aipa\\src')
from optimisers.activation_fn import sigmoid, sigmoid_derivative
from backpropagation.gradient_descent import mini_batch_gradient_descent
from preprocess.word_embedder import bag_of_words, embed_intents

def train_model(model: lin_model):
    intents: dict = get_intents()
    inputs = load_training_data()
    bow = bag_of_words(inputs[:,0])
    vectorised_x = bow.get_embeddings(inputs[:,0])
    vectorised_y = embed_intents(list(intents.keys()), inputs[:,1])
    preprocessed_inputs = [(vectorised_x[i], vectorised_y[i]) for i in range(len(vectorised_y))]
    print(vectorised_y)
    model.add_layer(sigmoid, sigmoid_derivative)
    model.add_layer(sigmoid, sigmoid_derivative)
    model.add_layer(sigmoid, sigmoid_derivative)
    model.add_layer(sigmoid, sigmoid_derivative, 3) #output shape: [y1 (reminders intent),y2 (email intent), y3 (other intent)], y1 & y2 b/w 0-1 inclusive
  
    # mini_batch_gradient_descent(model, [([10,20,10,20,30], [1,0,0])], [([10,20,10,40,2],[1,0,0])],test_function=test_model,batch_size=1)
    mini_batch_gradient_descent(model, preprocessed_inputs, [([10,20,10,40,2],[1,0,0])],test_function=test_model,batch_size=5)

# samples shape: list[(input vector: list[float], actualOutput: list[float])], actualOutput is the numerical rep of intents
def test_model(model: lin_model, samples, prefix:str):
    avg_acc = 0
    for i, sample in enumerate(samples):
        output = model.feed_forward(sample[0], cleanAfter=True)
        print('output:',output, 'actual:',sample[1])
        errs = np.subtract(output, sample[1])
        avg_acc += 1-np.true_divide(np.dot(errs,errs),len(errs))
    avg_acc /= len(samples)
    avg_acc *= 100
    print(prefix,'Test accuracy: ',avg_acc,'%')

def load_training_data():
    df = pd.read_csv("src/intents/dataset/aipa_dataset.csv")
    
    input_arr = df.to_numpy()
    random.shuffle(input_arr)

    print(len(input_arr))
    
    return input_arr
    #print(df.get(['statement']))


def get_intents():
    intents_fv = open('src/intents/intents.json', "r")
    intents_input = intents_fv.read()
    intents_fv.close()
    return json.loads(intents_input)

train_model(lin_model(input_size=159,layer_size=120, learning_rate=0.01))



