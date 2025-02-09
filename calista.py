# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  This file serves as the main file for Calista. This file will handle
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

# Tokenize, Pad and Normalize the training data
norm_x = tokenizer.tokenize_pad_normalize(tokenizer, training_data, 0)
norm_y = tokenizer.tokenize_pad_normalize(tokenizer, training_data, 1)

# Output needs to be one-hot encoded. Or better yet it can be a fixed length and we pad the output

#-----------------------------------------------------------------------------#
# Create the AI models that will be used and train them on the training data. #
#-----------------------------------------------------------------------------#

# The input sizes will depend on the size of the X data in the training data.
predictive_obj = Predictive_NN(input_size=353, hidden_size=1024, hidden2_size=1024, output_size=2)
transformer_obj = Transformer(vocab_size=500, embed_size=32, num_heads=4, num_layers=2, feedforward_dim=64)

X_predicted_tokens = predictive_obj.forward(norm_x[1])
# Need to create a map to the specific tasks that the model predicts.
# Any value within a limit close to 1 will be ran. Any value less than limit will be
# valued at 0 and will not run. The specific task to run is based on the position of
# the values within the models output. The limit is hard coded but we can also create
# a value that gets adjusted based on the models accuracy of the tasks performed.
# Find places to create values within the models to train and adjust based on accuracy
# of all the models. Step by step we will define and create the network we desire.

# Almost need two different expected outputs. One specifically for the predictive model
# and the other for the expected output from the transformer model.


predicted_tokens = [5, 12] # Should be generated after each input from the user.
print(f'Predictive Model output: \n{X_predicted_tokens}\n')
#usr_input_tokens = norm_x[1]
usr_input_tokens = [20, 8]

seed = np.concatenate((predicted_tokens, usr_input_tokens))
#seed = np.concatenate((predicted_tokens[0], usr_input_tokens)) # What we are feeding into the transformer object

#output = generate_text(transformer_obj, seed, length=len(norm_y[0]), vocab_size=500)
print(f'Transformer Model Steps:')
output = generate_text(transformer_obj, seed, length=len(norm_y[0]), vocab_size=500)
print(f'Seed: \n{list(seed)}\n')
print(f'Generated Transformer Output: \n{output}')
print(f'Predictive Model Output: \n{X_predicted_tokens}')

print(logo)