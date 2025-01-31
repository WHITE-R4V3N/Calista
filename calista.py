# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  This file serves as the main file for IRIS. This file will handle
#               user input and give the neural network output in response. This includes
#               running tests and or commands. All will be kept in a log file for documentation
#               purposes. Name based on greek goddess for communication.

import base64
import string

from data_parse import *
from models_obj import *
from settings import *
from logs.logger import *

nn_visualizer = False
version = 'v 2.0.0'

logo = f'''
\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------{RESET} {version}
\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Capstone project and only to be used for ethical purposes!
'''

parsed_files = ''

#-----------------------------------------------#
# Create the tokenizer and load the data files. #
#-----------------------------------------------#
tokenizer = DataTokenizer()

tokenizer.add_data_file(json.load(open('datasets/cryptography_data.json', 'r')))
tokenizer.add_data_file(json.load(open('datasets/command_data.json', 'r')))

# Parse the files
training_data = tokenizer.parse_files()

# Now we need to tokenize each string of data
print(training_data)
X = []
for i in training_data:
    print(f'{training_data[i][0]}\n{training_data[i][1]}')

    data_x = ''

    for labels in training_data[i][0]:
        # Labels are used to pull the data needed. This way we only need one function rather than
        # Hard coding each and every field for all training files.
        data_x += f'{str(training_data[i][0][labels])} '

    X.append(tokenizer.char_tokenize(data_x))
    X_tokens = tokenizer.char_tokenize(data_x) # Tokenize the X data
    #X_pad_input = tokenizer.pad_input(X_tokens)
    # Normalize the data

    tokenizer.max_length = max(len(seq) for seq in X)
    print(f'{X_tokens}\n\n')

padded_x = tokenizer.pad_input(X)
# Noramlize the X data now

#-----------------------------------------------------------------------------#
# Create the AI models that will be used and train them on the training data. #
#-----------------------------------------------------------------------------#

# The input sizes will depend on the size of the X data in the training data.
predictive_obj = Predictive_NN(input_size=64, hidden_size=128, hidden2_size=128, output_size=1)
transformer_obj = Transformer(vocab_size=100, embed_size=32, num_heads=4, num_layers=2, feedforward_dim=64)

# Need to create the input tokens here
predicted_tokens = [5, 12]
usr_input_tokens = [20, 8]
seed = predicted_tokens + usr_input_tokens

output = generate_text(transformer_obj, seed, length=10, vocab_size=100)
print(f'\nGenerated commands: \n{output}')