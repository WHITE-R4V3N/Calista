#   Description:    Parse the json data files used for training and pass to the tokenizer.

import json
import numpy as np

from collections import defaultdict

# A variable to make adding a specific flag prefix to the datasets
flag_prefix = 'CTF'

class DataTokenizer:
    def __init__(self):
        self.word_index = defaultdict(lambda: len(self.word_index))
        self.max_length = 0
        self.data = []
        self.corpus = {}

    def add_data_file(self, file_data):
        self.data.append(file_data)

    # Convert each letter or number to its ASCII value
    def char_tokenize(self, text):
        return [ord(char) for char in text]
    
    # Pads the X input for uniform data size
    def pad_input(self, data_pad):
        self.max_length = max(len(seq) for seq in data_pad)

        return [seq + [0] * (self.max_length - len(seq)) for seq in data_pad]
    
    # Normalizes the values (creates a decimal 0 = 0 and 1 = 65535)
    def normalize_array(self, digi, max_value):
        return digi / max_value
    
    # Parses the inputs and outputs from the data file given
    def parse_files(self):
        preparsed_data = []

        for data in self.data:
            for key in data:
                for item in data[key]:
                    preparsed_data.append(item)
        
        id_value = 0
        parsed_data = {}

        for item in preparsed_data:
            parsed_data[id_value] = [item['inputs'], item['outputs']]
            id_value += 1

        return parsed_data
    
    def tokenize_data(self, tokenizer, training_data, x_or_y):
        if x_or_y == 0:
            tokenized_data = []
            for input_output in training_data:
                d_str = ''
                for labels in training_data[input_output][x_or_y]: # The var changning the data changes this 0 and the 0 below to a variable instead
                    d_str += f'{training_data[input_output][x_or_y][labels]} '

                tokenized_data.append(tokenizer.char_tokenize(d_str))
            
            return tokenized_data
        else:   # Data is equal to the training models output. For predictive we need to convert to a corpus array
                # For the transformer model we need to do all of the steps.
            tokenized_transformer = []
            tokenized_predictive = []

            for input_output in training_data:
                d_str = ''

                for labels in training_data[input_output][1]['transformer']:
                    d_str += f'{training_data[input_output][1]['transformer'][labels]} '
                tokenized_transformer.append(tokenizer.char_tokenize(d_str))

                # For predictive d_str is more of a d_arr as we need to create a corpus rather than tokenizing and normalizing the data.
                d_arr = {}
                for labels in training_data[input_output][1]['predictive']:
                    d_arr[labels] = (training_data[input_output][1]['predictive'][labels])
                tokenized_predictive.append(d_arr)

            return tokenized_transformer, tokenized_predictive

    def pad_normalize_data(self, tokenized_data):
        padded_data = self.pad_input(tokenized_data)

        # Can be generalized to just return normalized_data
        normalized_data = []
        for item in padded_data:
            data_max = max(i for i in item)
            digi_arr = []

            for digi in item:
                digi_arr.append(self.normalize_array(digi, data_max))
            
            normalized_data.append(digi_arr)

        return normalized_data
    
    def construct_corpus(self, y_pred_tok): # This creates a corpus for the predictive model y training data
        position = 0

        for i in range(len(y_pred_tok)):
            for key in y_pred_tok[i]:
                if key not in self.corpus:
                    self.corpus[key] = position
                    position += 1

        # Now convert each y_pred_tok to corpus data or norm_predictive
        norm_predictive = []

        for j in range(len(y_pred_tok)):
            d_arr = []
            for key in self.corpus:
                if key in y_pred_tok[j]:
                    d_arr.append(1)
                else:
                    d_arr.append(0)
            
            norm_predictive.append(d_arr)

        return norm_predictive