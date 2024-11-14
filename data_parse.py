#   Description:    Parse the json data files used for training and pass to the tokenizer.

import json
import numpy as np

from collections import defaultdict

# Load the file
json_data = json.loads(open('datasets/json_training_data.json', 'r').read())

categories = []
algorithm_cipher = {}
challenge = []
flags = {}

flag_prefix = ''    # Can set a specific prefix the model can look for in flags. Train the AI at program start.

# Class to tokenize the data being given
class DataTokenizer:
    def __init__(self):
        self.word_index = defaultdict(lambda: len(self.word_index))
        self.max_length = 0

    def fit_on_texts(self, texts):
        for text in texts:
            for word in text.split():
                self.word_index[word]

    # Used for sentence processing or hint processing
    def texts_to_sequence(self, texts):
        return [[self.word_index[word] for word in text.split()] for text in texts]
    
    # Convert each letter to ASCII value
    def char_tokenize(self, text):
        return [ord(char) for char in text]

    # Function to pad the text or data
    def pad_input(self, data_pad):
        self.max_length = max(len(seq) for seq in data_pad)

        return [seq + [0] * (self.max_length - len(seq)) for seq in data_pad]
    
    def normalize_array(arr, max_value=65535):
        return arr / max_value

for entry in json_data['cryptography']:
    #categories.append[entry]

    algorithm_cipher[entry['ciphertext']] = entry['algorithm']
    challenge.append(entry['hint'])

for entry in json_data['flag_identification']:
    flags[entry['sequence']] = entry['contains_flag']