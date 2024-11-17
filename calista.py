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

logo = f'''
\n\n\n\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------{RESET}\tv 1.0.5
\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Capstone project and only to be used for ethical purposes!
'''

create_file() # Creates the files for the logs

# Create an object for the neural network that predicts what cipher is used on a piece of text
cipher_model = Base_Model(Simplified_NN(input_size=72, hidden_size=256, output_size=5),
                          DataTokenizer(json.loads(open('datasets/json_training_data.json', 'r').read())),
                          json.loads(open('datasets/json_training_data.json', 'r').read()))
                          # Last item here is redundant. Tokenizer has it saved as data_file. Use self.tokenizer.data_file

# These happen for each model and may be able to be moved into a function but for now will be left alone.
algorithm_cipher, challenge = cipher_model.tokenizer.parse_cypto()      # Parses relevant information needed for cypto analysis
cipher_model.create_X_y_training(algorithm_cipher)                      # Parses and creates the training data used for the neural network
final_loss = cipher_model.model.train(cipher_model.X, cipher_model.y)   # Trains the network and returns the final loss number
append_data(f'Network is done loading cipher model. Final loss is {final_loss}.\n')  # Appends the loss to the log file for reference and analyzing what happened in logs

flag_model = Base_Model(Simplified_NN(input_size=60, hidden_size=128, output_size=2),
                        DataTokenizer(json.loads(open('datasets/json_training_data.json', 'r').read())),
                        json.loads(open('datasets/json_training_data.json', 'r').read()))

flag_dict = flag_model.tokenizer.parse_flag()
flag_model.create_X_y_training(flag_dict)
final_loss = flag_model.model.train(flag_model.X, flag_model.y)
append_data(f'Network is done loading flag identification model. Final loss is {final_loss}.\n') # Appends the loss to the log file for reference and analyzing what happened in logs

#--------------------------------------------------------------------------------
#   The loop for taking in user input and or questions, flags, ciphers, etc.    |
#--------------------------------------------------------------------------------
print(f'{logo}\n{CAL_COL}Calista{RESET}> Hi, I am Calista. I am a work in progress AI that will compete in\nCapture the flag competitions.\n')

while True:
    usr_prompt = input(f'{USER}User{RESET}> ')

    append_data(f'*USER INTERACTION*\n\nUser has entered:\n{usr_prompt}\n')

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):
        print('\n')
        exit(-1)
    else:
        x_usr = [cipher_model.tokenizer.char_tokenize(usr_prompt)]
        x_usr = [seq + [0] * (cipher_model.tokenizer.max_length - len(seq)) for seq in x_usr]
        x_usr = (x_usr - cipher_model.X_min) / (cipher_model.X_max - cipher_model.X_min)

        append_data(f'Normalized user prompt:\n{x_usr}\n')

        prediction = cipher_model.model.predict(x_usr)
        append_data(f'Model Prediction: {prediction}({cipher_model.labels_index[prediction[0]]})\n')
        print(f'\nCipher Type: {GREEN}{cipher_model.labels_index[prediction[0]]}{RESET}\n\n{YELLOW}Attempting to retrieve flag...{RESET}\n')

        try:
            if prediction[0] == 4:                          # Decrypt base64 encryption
                usr_decode = base64.b64decode(usr_prompt)
                usr_plain = usr_decode.decode('ascii')
            elif prediction[0] == 0:
                pass
                usr_decode = usr_prompt.strip().lower()
                key = 5         # For now hard coded to 5. Will either predict key or just go -26 to 26
                alphabet = string.ascii_lowercase
                usr_plain = ''

                # Attempt to decipher, send to flag identifying model. If no flag keep going. Else print deciphered flag

                for c in usr_decode:
                    if c in alphabet:
                        position = alphabet.find(c)
                        new_position = (position - key)
                        new_character = alphabet[new_position]
                        usr_plain += new_character

            elif prediction == 3:
                usr_decode = usr_prompt.split(' ')          # Decrypt A1Z26 encryption
                usr_plain = "".join(chr(int(elem) + 64) for elem in usr_decode)

            append_data(f'Cipher text to plain text:\n{usr_prompt} -> {usr_plain}\n')
            print(f'{RED}{usr_prompt}{RESET} -> {GREEN}{usr_plain}{RESET}\n')
        except:
            print(f'Something went wrong while computing.\nPrediction: {RED}{prediction[0]}{RESET} ({YELLOW}{cipher_model.labels_index[prediction[0]]}{RESET})\n')
            append_data(f'Something went wrong with the model\'s prediction.\nPrediction was:\n{prediction}({cipher_model.labels_index[prediction[0]]})\n')