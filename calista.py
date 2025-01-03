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

nn_visualizer = True
version = 'v 1.0.10'

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

create_file() # Creates the files for the logs

#####       ALL MODELS BEING USED     #####
print(f'\n{YELLOW}Loading Cipher Model:{RESET}')
# Create an object for the neural network that predicts what cipher is used on a piece of text
cipher_model = Base_Model(Simplified_NN(input_size=72, hidden_size=256, output_size=5),
                          DataTokenizer(json.loads(open('datasets/json_training_data.json', 'r').read())),
                          json.loads(open('datasets/json_training_data.json', 'r').read()))
                          # Last item here is redundant. Tokenizer has it saved as data_file. Use self.tokenizer.data_file

# These happen for each model and may be able to be moved into a function but for now will be left alone.
cipher_dict = cipher_model.tokenizer.parse_cypto()      # Parses relevant information needed for cypto analysis
final_loss = create_model(cipher_model, cipher_dict)
append_data(f'Network is done loading cipher model. Final loss is {final_loss}.\n')  # Appends the loss to the log file for reference and analyzing what happened in logs

print(f'\n{YELLOW}Loading Flag Identification Model:{RESET}')
flag_model = Base_Model(Simplified_NN(input_size=60, hidden_size=128, output_size=2),
                        DataTokenizer(json.loads(open('datasets/json_training_data.json', 'r').read())),
                        json.loads(open('datasets/json_training_data.json', 'r').read()))

flag_dict = flag_model.tokenizer.parse_flag()
final_loss = create_model(flag_model, flag_dict)

# If final loss is greater than 0.01 then re-train the model until final_loss is less than 0.01
append_data(f'Network is done loading flag identification model. Final loss is {final_loss}.\n') # Appends the loss to the log file for reference and analyzing what happened in logs

print(f'\n{YELLOW}Loading Command Model:{RESET}')
# Create an object for the neural network that predicts what cipher is used on a piece of text
command_model = Base_Model(Simplified_NN(input_size=72, hidden_size=1028, output_size=5),
                          DataTokenizer(json.loads(open('datasets/json_training_data.json', 'r').read())),
                          json.loads(open('datasets/json_training_data.json', 'r').read()))
                          # Last item here is redundant. Tokenizer has it saved as data_file. Use self.tokenizer.data_file

# These happen for each model and may be able to be moved into a function but for now will be left alone.
command_dict = command_model.tokenizer.parse_cypto()      # Parses relevant information needed for cypto analysis
final_loss = create_model(command_model, command_dict)
append_data(f'Network is done loading command generating model. Final loss is {final_loss}.\n')

#######################################################

#--------------------------------------------------------------------------------
#   The loop for taking in user input and or questions, flags, ciphers, etc.    |
#--------------------------------------------------------------------------------
print(f'{logo}\n{CAL_COL}Calista{RESET}> Hi, I am Calista. I am a work in progress AI that will compete in\nCapture the flag competitions.\n')

# Continous loop to get user input
while True:
    usr_prompt = input(f'{USER}User{RESET}> ')                              # User input as raw text

    append_data(f'*USER INTERACTION*\n\nUser has entered:\n{usr_prompt}\n') # Sending the user input to the logs

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):       # Quit the program if user enters q or quit
        print('\n')
        exit(-1)
    else:
        x_usr = [cipher_model.tokenizer.char_tokenize(usr_prompt)]                              # Tokenize each character of the user input
        x_usr = [seq + [0] * (cipher_model.tokenizer.max_length - len(seq)) for seq in x_usr]   # Create padding for the user input
        x_usr = (x_usr - cipher_model.X_min) / (cipher_model.X_max - cipher_model.X_min)        # Normalize the user input data

        append_data(f'Normalized user prompt:\n{x_usr}\n')                                      # Add normalized user input to logs

        prediction = cipher_model.model.predict(x_usr)                                          # Get a prediction back from the cipher model
        append_data(f'Model Prediction: {prediction}({cipher_model.labels_index[prediction[0]]})\n')
        print(f'\nCipher Type: {GREEN}{cipher_model.labels_index[prediction[0]]}{RESET}\n\n{YELLOW}Attempting to retrieve flag...{RESET}\n')
        # Add the prediction to the log file and print the info back to the user

        try:
            if prediction[0] == 4:                          # Decrypt base64 encryption
                usr_decode = base64.b64decode(usr_prompt)
                usr_plain = usr_decode.decode('ascii')      # Print as plain text to the user
            elif prediction[0] == 0:
                pass
                usr_decode = usr_prompt.strip().lower()
                key = 9         # For now hard coded to 5. Will either predict key or just go -26 to 26
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
                usr_decode = usr_prompt.split(' ')                                  # Decrypt A1Z26 encryption
                usr_plain = "".join(chr(int(elem) + 64) for elem in usr_decode)

            # Append the models accurate prediction and/or decryption to the logs
            append_data(f'Cipher text to plain text:\n{usr_prompt} -> {usr_plain}\n')
            print(f'{RED}{usr_prompt}{RESET} -> {GREEN}{usr_plain}{RESET}\n')


            # SUDO CODE
            # 1. Tokenize the output of decrypted text (usr_plain)
            x_flag = [flag_model.tokenizer.char_tokenize(usr_plain)]
            # 2. Pad the tokenized output of decrypted text
            x_flag = [seq + [0] * (flag_model.tokenizer.max_length - len(seq)) for seq in x_flag]
            # 3. Normalize the padded data
            x_flag = (x_flag - flag_model.X_min) / (flag_model.X_max - flag_model.X_min)
            # 3. Send the data to the flag_model for flag identification.
            f_prediction = flag_model.model.predict(x_flag)

            append_data(f'Flag Model Prediction: {prediction}\n')

            if f_prediction:
                print(f'{GREEN}[+]{RESET} Output contains flag.\n')
            else:
                print(f'{RED}[-]{RESET} Output does not contain a flag.\n')
        except:
            # When something goes wrong add the failed prediction to the logs
            print(f'Something went wrong while computing.\nPrediction: {RED}{prediction[0]}{RESET} ({YELLOW}{cipher_model.labels_index[prediction[0]]}{RESET})\n')
            append_data(f'Something went wrong with the model\'s prediction.\nPrediction was:\n{prediction}({cipher_model.labels_index[prediction[0]]})\n')

        if nn_visualizer: # If the visualizer variable is set to true then display the nerual network
            cipher_model.model.visualize(np.array(x_usr)) # Create a visual of the model