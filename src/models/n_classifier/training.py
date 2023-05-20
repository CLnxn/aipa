import pandas as pd
import json
import nltk
from nltk.stem import WordNetLemmatizer
from classifier import lin_model
import random
import sys
sys.path.insert(0, 'C:\\Users\\proki\\repos\\aipa\\src')
from optimisers.activation_fn import sigmoid, sigmoid_derivative


lemmatizer = WordNetLemmatizer()
def train_model(model: lin_model):
    intents = get_intents()
    inputs = load_training_data()
    model.add_layer(sigmoid, sigmoid_derivative)
    model.add_layer(sigmoid, sigmoid_derivative)
    model.add_layer(sigmoid, sigmoid_derivative, 2)
    output = model.feed_forward(input=[10,20,10])
    print('output:',output)

    





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

train_model(lin_model())




