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

tokenizer.add_data_file(json.loads(open('datasets/cryptography_data.json', 'r').read()))
tokenizer.add_data_file(json.loads(open('datasets/command_data.json', 'r').read()))

# Now we need to tokenize each string of data

#-----------------------------------------------------------------------------#
# Create the AI models that will be used and train them on the training data. #
#-----------------------------------------------------------------------------#

# The input sizes will depend on the size of the X data in the training data.
predictive_obj = Predictive_NN(input_size=64, hidden_size=128, hidden2_size=128, output_size=1)
