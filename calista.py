# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  This file serves as the main file for Calista. This file will handle
#               user input and give the neural network output in response. This includes
#               running tests and or commands. All will be kept in a log file for documentation
#               purposes. Name based on greek goddess for communication.

# This would be really cool if it developed into a tool used to help develop Red or Blue team skills in a CTF enviroment.
# A CTF powered by AI where you eather break into and get flags off a system with an AI blue team trying to stop you.
# or you try to defend a CTF system from Red team AI and protect the flags. I think this would be a really cool idea that has not been donbe before.
# S-AI or Security Artificial (Algorithmic) Intelligence

import time
import multiprocessing

from data_parse import *
from models_obj import *
from settings import *
from logs.logger import *

from scripts.web_scrape import *

nn_visualizer = False
version = 'v 2.0.2'

logo = f'''
\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------{RESET} {BLUE}{version}{RESET}
\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Calista is a capstone project and only to be used for ethical purposes!
'''

tokenizer = DataTokenizer()

tokenizer.add_data_file(json.load(open('datasets/cryptography_data.json', 'r')))
tokenizer.add_data_file(json.load(open('datasets/command_data.json', 'r')))

# Parse the data files
training_data = tokenizer.parse_files()

# Tokenize, Pad and normalize the respective data
tok_x = tokenizer.tokenize_data(tokenizer, training_data, 0)
pad_tok_x = tokenizer.pad_input(tok_x)
norm_x = tokenizer.pad_normalize_data(tok_x)

tok_transformer_y, tok_predictive_y = tokenizer.tokenize_data(tokenizer, training_data, 1)      # This is fine as we can get it to return a list for each expected y
norm_transformer = tokenizer.pad_input(tok_transformer_y)                                       # Does not need to be normalized but will need to be padded
norm_predictive = tokenizer.construct_corpus(tok_predictive_y)                                  # Creates the corpus to be used and converts the predictive y to corpus data

# Not sure if I need to tokenize or normalize. For each item set the value to 1 or 0 based on one-hot encoding????????????

# The input sizes will depend on the size of the X data in the training data. And output is based on norm_predictive item length
predictive_obj = Predictive_NN(input_size=345, hidden_size=1024, hidden2_size=1024, output_size=len(norm_predictive[0]))
transformer_obj = Transformer(vocab_size=1000, d_model=64, seq_len=10, num_heads=8, learning_rate=0.01)

# Create and add the data to the log file
create_file()
append_data(f'Tokenizer:\n{tokenizer}')
append_data(f'Training Data:\n{training_data}')
append_data(f'Normalized X:\n{norm_x}')
append_data(f'Normalized Transformer y:\n{norm_transformer}')
append_data(f'Normalized Predictive y:\n{norm_predictive}')
append_data(f'Predictive Object:\n{predictive_obj}')
append_data(f'Transformer Object:\n{transformer_obj}')

# Train the predictive model
print(f'\nLoading Predictive Model:')
training_continue = True

while training_continue:
  network_loss = 'NULL'
  network_loss = predictive_obj.train(np.array(norm_x), np.array(norm_predictive))        # Train the predictive network
  append_data(f'Predictive network iteration losses: \n{network_loss}')

  print(f'Final Predictive Network Loss: {network_loss[-1]}')

  # Test Predictive model
  X_predicted_tokens = predictive_obj.forward(norm_x[1])
  predicted_tokens = [round(tokens) for tokens in X_predicted_tokens[0]]

  if predicted_tokens == norm_predictive[1]:
    print(f'TEST PREDICTION {GREEN}SUCCESS!{RESET}')
    append_data(f'TEST PREDICTION SUCCESS!')
    training_continue = False
  else:
    print(f'TEST PREDICTION {RED}FAILED!{RESET}')
    append_data(f'TEST PREDICTION FAILED!')
    # Train again

# Train the Transformer model
print(f'\nLoading Transformer Model:')
transformer_x = []      # This will be the seeded X data
for i in range(len(pad_tok_x)):
  transformer_x.append(np.concatenate((norm_predictive[i], pad_tok_x[i])))

append_data(f'Transformer X:\n{transformer_x}')

#network_loss = transformer_obj.train(np.array(list(transformer_x)), norm_transformer)
append_data(f'Transformer network iteration losses: \n{network_loss}')
print(f'Final Transformer Network Loss: {network_loss[-1]}')

# Test Transformer model
# Call training function

usr_input_tokens = pad_tok_x[1]          # Will need to be padded for this to work
seed = np.concatenate((predicted_tokens, usr_input_tokens))
output = transformer_obj.generate_command(seed) # Length should be adjustable for y

# Check for correct token generation

append_data(f'X predictive tokens:\n{X_predicted_tokens}')
append_data(f'User input tokens:\n{usr_input_tokens}')
append_data(f'Seed: \n{seed}')
append_data(f'Transformer output tokens:\n{output}\nOutput Shape: {np.array(output).shape}')

#ctf_url = input(f'\n\n{YELLOW}Please enter the CTF challenge url here: {RESET}')
#page = get_challeneges(ctf_url)
#append_data(f'TESTING CTF WEBPAGE GRABBING: \n{page}')

print(logo)

# Create a multiprocessing sequence to process predictive, transformer and 
# visualize the networks in real time together. This will help increase the
# speed of Calista and lower computation times making the process much faster.

# Process 1: Predictive model runs, then transformer runs
# Process 2: Create the visual or rather update the visuals (for both models)

#   or

# Process 1: Predictive model training data
# Process 2: Transformer model training data
# Process 3: Visualize the network being trained

# This will make it much faster in the training process.

queue = multiprocessing.Queue()