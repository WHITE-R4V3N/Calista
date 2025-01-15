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
        self.data_files = []

    def add_data_file(self, file_data):
        self.data_files.append(file_data)

    # Convert each letter or number to its ASCII value
    def char_tokenize(self, text):
        return [ord(char) for char in text]
    
    # Pads the X input for uniform data size
    def pad_input(self, data_pad):
        self.max_length = max(len(seq) for seq in data_pad)

        return [seq + [0] * (self.max_length - len(seq)) for seq in data_pad]
    
    # Normalizes the values (creates a decimal 0 = 0 and 1 = 65535)
    def normalize_array(arr, max_value=65535):
        return arr / max_value
    
    # Parses the inputs and outputs from the data file given
    def parse_files(self):
        # How can I pull the dictionary values from the JSON data??
        # That way I do not need to hard code anything for parsing files.
        # Should I just convert the data into a dictionary from json after
        # reading it from a file?
        caesar_cipher = [item for item in [entry for entry in self.data_file['caesar_cipher']]]

        id_value = 0
        parsed_data = {}

        for item in caesar_cipher:
            parsed_data[id_value] = [item['inputs'], item['outputs']]
            id_value += 1

        return parsed_data
