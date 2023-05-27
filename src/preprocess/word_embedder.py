from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import deque
lemmatize = WordNetLemmatizer()
ignore_chars = set(['?', '!', ';', '*', '.'])
def embed_inputs( text_datas: list[list]):
    """
    takes in array of text input and converts them into an array of their numerical vector representations
    """
    for i, data_entry in enumerate(text_datas):
        reduce_sentence(data_entry[0])

def reduce_sentence(sentence:str):
    """
    Tokenizes, lemmatises & then splits a string sentence into array format
    """
    tokenized = word_tokenize(sentence)
    lemmatized = deque()
    for token in tokenized:
        lem_token = lemmatize.lemmatize(token.lower())
        lemmatized.append(lem_token) if lem_token not in ignore_chars else None
    return lemmatized
     
def embed_intents(intents: list[str], input_intents: list[list[str] | str]) -> list[list[int]]:
    """
        Converts an array of intents into a 1d binary vector representation \n
        Args:
            intents: default 1d array of unique intents, identified in positional order
            input_intents: an array of a document's intent which can either contain multiple intents (a list), or just 1 (str)
        Returns:
            A 1d array of 0s & 1s, 0 if it is not the intent at that position, 1 otherwise
    """
    intents = {intent:i for i, intent in enumerate(intents)}
    intents_vect = []
    for entry in input_intents:
        entry_intents = [0]*len(intents)
        if entry is list:
            
            for intent in entry:
                entry_intents[intents[intent]] = 1
            
            intents_vect.append(entry_intents)
            continue
        entry_intents[intents[entry]] = 1
        intents_vect.append(entry_intents)
    return intents_vect
def embed_methods():
    pass

class bag_of_words:
    def __init__(self, inputs: list[str]):
        self.inputs = inputs
        self.vocab = set()
        self.vocab_map = {} 
        self.internal_create_vocab()      
        print('VOCAB SIZE:',len(self.vocab))  

    def internal_create_vocab(self):
        vocab = set()
        for sentence in self.inputs:
            lemmatized_set = set(reduce_sentence(sentence)) # str-> list[str]
            vocab = vocab.union(lemmatized_set) 
        self.vocab_map = {word:i for i, word in enumerate(vocab)} # word string key -> standardized position index value
        self.vocab = vocab # lookup table

    def update_vocab(self, new_inputs: list[str], append=True):
        if append:
            return self.internal_create_vocab(new_inputs)
        
        for sentence in new_inputs:
            lemmatized_set = set(reduce_sentence(sentence))
            for word in lemmatized_set:
                if word not in self.vocab:
                    self.vocab.add(word)
                    self.vocab_map[word] = len(self.vocab)-1
        
    def get_embeddings(self, inputs: list[str] = None)-> list[list[int]]:
        inputs = self.inputs if inputs is None else inputs
        bow_inputs = []
        for input_pattern in inputs:
            lemmatized = reduce_sentence(input_pattern)
            bow = [0]*len(self.vocab)
            for word in lemmatized:
                if word in self.vocab:
                    bow[self.vocab_map[word]] += 1
            bow_inputs.append(bow)
        return bow_inputs
    
# b_o_w = bag_of_words(['I need a reminder to pay the bills by the end of the week.','Remind me to call John at 4:00 PM today.'])
# print(b_o_w.get_embeddings())