# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  The tokenizer trained for being used with Calista AI model.

#----------------
#    IMPORTS    |
#----------------
import numpy
import time
import regex as re
import codecs

from datasets import load_dataset

#-----------------------------------
#   The training data to be used   |
#-----------------------------------
file_paths = ["data/test1.txt", "data/test2.txt", "data/test3.txt", "data/test4.txt", "data/test5.txt", "data/test6.txt"]

ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
port_pattern = r'\b\d{1,5}\b'
flag_pattern = r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}"

corpus = {}
#----------------------------------------------------------------
#   Functions to tokenize the text or data and create a corpus  |
#----------------------------------------------------------------

def tokenize_data(text, tokens):
    # Tokenize the ip, port and flags within the data
    matches = re.findall(ip_pattern + '|' + flag_pattern + '|' + port_pattern, text)
    for match in matches:
        tokens.add(match)                           # Add the matched items to the tokens
        text = re.sub(ip_pattern, '<IP>', text)     # Remove the ip address and replace with another tag
        text = re.sub(flag_pattern, '<FLAG>', text) # Replace the flag if any with a tag
        text = re.sub(port_pattern, '<PORT>', text) # Replace the port with a tag

    # Tokenize the other words in the document
    words = re.findall(r'\b\w+\b', text)    # Create a token of every word including the new tags
    tokens.update(words)

    return tokens

def create_corpus(tokens):                  # Create a corpus which has is: token: (position, token)
    for i, token in enumerate(tokens):
        corpus[token] = (i, token)

    return corpus

def create_bin_array(corpus, text):
    tokens = set()
    binary_array = [0] * len(corpus) # Initialize array with zeros

    tokens = tokenize_data(text, tokens)     # Tokenize the text

    # Set array values to 1 for tokens found in the corpus
    for token in list(tokens):
        if token in corpus:
            index = corpus[token][0]
            binary_array[index] = 1

    return binary_array

# To be run at the start of the program to create a list of predefined words.
def train_corpus():
    tokens = set()
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            tokens = tokenize_data(text, tokens)

    corpus = create_corpus(tokens)  # Generated from all the data files to create a corpus to use
    print(f'{corpus}\n')

#create_bin_array(corpus, text) # This will be the text from the user

# Lets see if this fixes the problem we are having.
print('Training the tokenizer:')
train_corpus()