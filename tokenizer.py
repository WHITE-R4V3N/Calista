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

file_paths = ["data/test1.txt", "data/test2.txt", "data/test3.txt"]

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

        # Create specific tokens for flags, ip_address, ports
        text_token = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', text)
        text_token = re.sub(r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}", '<FLAG>', text_token)

        ip_addresses = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
        flags = re.findall(r"\w+'?\w*\{[a-zA-Z0-9_'-]+\}", text)

        print(text_token)
        print(f'IP: {ip_addresses}')
        print(f'Flag: {flags}\n----------------------------------------------\n')

