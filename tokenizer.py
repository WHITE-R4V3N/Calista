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

file_paths = ["data/test1.txt", "data/test2.txt", "data/test3.txt", "data/test4.txt", "data/test5.txt"]

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

        # Create specific tokens for flags, ip_address, ports
        ip_addresses = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
        text_token = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)

        flags = re.findall(r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}", text_token)
        text_token = re.sub(r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}", '<FLAG>', text_token)

        port = re.findall(r'\b\d{1,5}\b', text_token)
        text_token = re.sub(r'\b\d{1,5}\b', '<PORT>', text_token)

        print(text_token)
        print(f'IP: {ip_addresses}')
        print(f'Port: {port}')
        print(f'Flag: {flags}\n----------------------------------------------\n')

        tokens = list(map(int, text_token.encode('utf-8')))
        print(tokens)


# Create a bunch of files based on Cyber security blog posts and along with many CTF challenges. This will fill out the
# possible tokens that can appear. We will then use this to make a corpus based on inputs. We can then send data to be
# tokenized and compared to the corpus. From there we can use this information tp get desired results from the server.
# We can do this by running the test data through the tokenizer and saving the corpus created. We can then use that
# to train the AI model and set the expected output. (ie. Scan and dirb, [1,0,0,0,0,0,1,...]).

