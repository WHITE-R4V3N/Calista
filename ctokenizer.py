# Author:       Emma Gillespie
# Date:         2025-05-20
# Description:  A tokenizer to convert the datasets into neural netwrok readable data.
import pickle
import hashlib
import re

from common import *

#----------------------------------------
#   Class for the tokenizer to be used  |
#----------------------------------------
class Tokenizer:
    def __init__(self, mode='word'):
        self.mode = mode
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.custom_tokens = {
            "<ip>": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "<email>": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            #"<python_file>": r'[a-zA-Z0-9-]+\.py',
            #"<domain>": r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b"
        }

        self.user_specials = {}

    def build_vocab(self, texts):
        #----------------------------------------
        #   Build the vocab for the datasets    |
        #----------------------------------------
        # This will now either tokenize the data by word or by individual character
        for text in texts:
            for token, pattern in self.custom_tokens.items():
                text = re.sub(pattern, token, text)

            # If len(text) > 10 and no space
            #   build vocab based on character (this should target the cryptographic stuff)
            #   minus a1z26..... hmmm we will need to think about this a little more then.

            for item in text.split():
                tokens = list(item.lower()) if self.mode == 'char' else item.lower().split()
                for token in tokens:
                    if token not in self.word2idx:
                        idx = len(self.word2idx)

                        self.word2idx[token] = idx
                        self.idx2word[idx] = token

    def encode(self, text, max_len):
        #------------------------
        #   Encode the dataset  |
        #------------------------
        for item in text:
            for token, pattern in self.custom_tokens.items():
                text = re.sub(pattern, token, text)

        tokens = list(text.lower()) if self.mode == 'char' else text.lower().split()

        token_ids = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
        token_ids = [self.word2idx['<sos>']] + token_ids + [self.word2idx['<eos>']]

        return token_ids[:max_len] + [self.word2idx['<pad>']] * (max_len - len(token_ids))
    
    def decode(self, token_ids):
        #------------------------
        #   Decode the dataset  |
        #------------------------
        tokens = [self.idx2word.get(tid, '<unk>') for tid in token_ids]
        return " ".join(t for t in tokens if t not in ("<pad>", "<sos>", "<eos>"))
    
    def parse_datasets(self, data):
        #------------------------
        #   Parse the dataset   |
        #------------------------
        dataset_data = []
        inputs, outputs = [], []

        for src, tgt in data:
            dataset_data.append(src)
            dataset_data.append(tgt)

            inputs.append(src)
            outputs.append(tgt)

        return dataset_data, inputs, outputs
