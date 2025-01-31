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
print(f'{training_data}\n')
'''
X = []
for i in training_data:
    print(f'{training_data[i][0]}\n{training_data[i][1]}') # Can remove

    data_x = ''

    for labels in training_data[i][0]:
        data_x += f'{str(training_data[i][0][labels])} '

    X.append(tokenizer.char_tokenize(data_x))
    X_tokens = tokenizer.char_tokenize(data_x)

    tokenizer.max_length = max(len(seq) for seq in X)
    print(f'{X_tokens}\n\n')                                # Can remove

padded_x = tokenizer.pad_input(X)
print(padded_x)
'''

parsed_x = []
for input_output in training_data:
    x_string = ''
    for labels in training_data[input_output][0]:
        x_string += f'{training_data[input_output][0][labels]} '

    parsed_x.append(x_string)

tokenized_x = []
for i in parsed_x:
    tokenized_x.append(tokenizer.char_tokenize(i))

padded_x = tokenizer.pad_input(tokenized_x)

normalized_x = []
for item in padded_x:
    x_max = max(i for i in item)
    normalized_x.append(tokenizer.normalize_array(item, x_max))

print(f'{parsed_x}\n')
print(f'{tokenized_x}\n')
print(f'{padded_x}\n')
print(f'{x_max}\n')
print(f'{normalized_x}\n')

# WHY THE FUCK IS NORMALIZING SO FUCKING HARD TO DO! LIKE FUCK

#-----------------------------------------------------------------------------#
# Create the AI models that will be used and train them on the training data. #
#-----------------------------------------------------------------------------#

# The input sizes will depend on the size of the X data in the training data.
predictive_obj = Predictive_NN(input_size=64, hidden_size=1024, hidden2_size=1024, output_size=1)
transformer_obj = Transformer(vocab_size=100, embed_size=32, num_heads=4, num_layers=2, feedforward_dim=64)

# Need to create the input tokens here
predicted_tokens = [5, 12]
usr_input_tokens = [20, 8]
seed = predicted_tokens + usr_input_tokens

output = generate_text(transformer_obj, seed, length=10, vocab_size=100)
print(f'\nGenerated commands: \n{output}')

print(logo)