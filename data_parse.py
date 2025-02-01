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
    
    def tokenize_pad_normalize(self, tokenizer, training_data, x_or_y):
        tokenized_data = []
        for input_output in training_data:
            x_string = ''
            for labels in training_data[input_output][x_or_y]: # The var changning the data changes this 0 and the 0 below to a variable instead
                x_string += f'{training_data[input_output][x_or_y][labels]} '

            tokenized_data.append(tokenizer.char_tokenize(x_string))

        if x_or_y == 0:
            padded_data = tokenizer.pad_input(tokenized_data)
        else:
            padded_data = tokenized_data

        # Can be generalized to just return normalized_data
        normalized_data = []
        for item in padded_data:
            data_max = max(i for i in item)
            digi_arr = []

            for digi in item:
                digi_arr.append(tokenizer.normalize_array(digi, data_max))
            
            normalized_data.append(digi_arr)

        return normalized_data