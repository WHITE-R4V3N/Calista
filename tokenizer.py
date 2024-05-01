# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  The tokenizer trained for being used with Calista AI model.

#----------------
#    IMPORTS    |
#----------------
import numpy
import time
import regex as re

from datasets import load_dataset

#-----------------------------------
#   The training data to be used   |
#-----------------------------------
file_paths = ["data/test1.txt", "data/test2.txt", "data/test3.txt", "data/test4.txt", "data/test5.txt"]

ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
port_pattern = r'\b\d{1,5}\b'
flag_pattern = r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}"

tokens = set()

#----------------------------------------------------------------
#   Functions to tokenize the text or data and create a corpus  |
#----------------------------------------------------------------
def tokenize_data(text):
    # Tokenize the ip, port and flags within the data
    matches = re.findall(ip_pattern + '|' + flag_pattern + '|' + port_pattern, text)
    for match in matches:
        tokens.add(match)
        text = re.sub(ip_pattern, '<IP>', text)
        text = re.sub(flag_pattern, '<FLAG>', text)
        text = re.sub(port_pattern, '<PORT>', text)

    # Tokenize the other words in the document
    words = re.findall(r'\b\w+\b', text)
    tokens.update(words)

    return tokens

def create_corpus(tokens):
    corpus = {}

    for i, token in enumerate(tokens):
        corpus[token] = (i, token)

    return corpus

def create_bin_array(corpus, text):
    binary_array = [0] * len(corpus) # Initialize array with zeros

    tokens = tokenize_data(text)     # Tokenize the text

    # Set array values to 1 for tokens found in the corpus
    for token in list(tokens):
        if token in corpus:
            index = corpus[token][0]
            binary_array[index] = 1

    return binary_array

# To be run at the start of the program to create a list of predefined words.
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        tokens = tokenize_data(text)

corpus = create_corpus(tokens)  # Generated from all the data files to create a corpus to use

with open("data/test5.txt", 'r', encoding='utf-8') as file:
    text = file.read()

    print(f'Tokens:\n{tokens}\n')
    tokens = set()

    binary_array = create_bin_array(corpus, text) # This will be the text from the user

    print(f'Corpus:\n{corpus}\n')
    print(f'Binary Array:\n{binary_array}\n')
    print(f'Text:\n{text}\n')

