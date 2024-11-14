# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  This file serves as the main file for IRIS. This file will handle
#               user input and give the neural network output in response. This includes
#               running tests and or commands. All will be kept in a log file for documentation
#               purposes. Name based on greek goddess for communication.

import base64

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
\t\t\t{YELLOW}-----------------------------{RESET}\tv 1.0.0
\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Capstone project and only to be used for ethical purposes!
'''

create_file()

#-------------------------------------------------------------------|
#   Creating the model object and training it using the json files  |
#-------------------------------------------------------------------|
cipher_labels = ['base64', 'caesar cipher', 'base64 -> base64 -> caesar cipher', 'rot13', 'A1Z26']
# Create a model object to predict cipher used. input size and output size will vary based on the data used
tokenizer = DataTokenizer()
cipher_dataset = json.loads(open('datasets/json_training_data.json', 'r').read())
cipher_model = Base_Model(Simplified_NN(input_size=72, hidden_size=128, output_size=5), tokenizer, cipher_dataset)

X = []
y = []
labels = []
cipher_labels = ['base64', 'caesar cipher', 'base64 -> base64 -> caesar cipher', 'rot13', 'A1Z26', '']
for ciphered_text in algorithm_cipher:
    X.append(cipher_model.tokenizer.char_tokenize(ciphered_text))
    X = cipher_model.tokenizer.pad_input(X)
    # Will eventually have to add a line to be able to normalize the Input data. For now this will work

    labels.append(algorithm_cipher[ciphered_text])

for label in labels:
    y.append(cipher_labels.index(label))

y = np.eye(5)[y] # one-hot encoding
X = np.array(X) # Normalize x before using. Causes overflow otherwise

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

# Train the NN for the cipher prediction model
final_loss = cipher_model.model.train(X, y)

append_data(f'Network is done loading. Final loss is {final_loss}.\n')

#--------------------------------------------------------------------------------
#   The loop for taking in user input and or questions, flags, ciphers, etc.    |
#--------------------------------------------------------------------------------
print(f'{logo}\n{CAL_COL}IRIS{RESET}> Hi, I am Calista. I am a work in progress AI that will compete in\nCapture the flag competitions.\n')

while True:
    usr_prompt = input(f'{USER}User{RESET}> ')

    append_data(f'*USER INTERACTION*\n\nUser has entered:\n{usr_prompt}\n')

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):
        print('\n')
        exit(-1)
    else:
        x_usr = [cipher_model.tokenizer.char_tokenize(usr_prompt)]
        x_usr = [seq + [0] * (cipher_model.tokenizer.max_length - len(seq)) for seq in x_usr]
        x_usr = (x_usr - X_min) / (X_max - X_min)

        append_data(f'Normalized user prompt:\n{x_usr}\n')

        prediction = cipher_model.model.predict(x_usr)
        append_data(f'Model Prediction: {prediction}({cipher_labels[prediction[0]]})\n')
        print(f'\nCipher Type: {GREEN}{cipher_labels[prediction[0]]}{RESET}\n\n{YELLOW}Attempting to retrieve flag...{RESET}\n')

        try:
            if prediction[0] == 0:                          # Decrypt base64 encryption
                usr_decode = base64.b64decode(usr_prompt)
                usr_plain = usr_decode.decode('ascii')
            elif prediction[0] == 1:
                pass
            elif prediction == 4:
                usr_decode = usr_prompt.split(' ')          # Decrypt A1Z26 encryption
                usr_plain = "".join(chr(int(elem) + 64) for elem in usr_decode)

            append_data(f'Cipher text to plain text:\n{usr_prompt} -> {usr_plain}\n')
            print(f'{RED}{usr_prompt}{RESET} -> {GREEN}{usr_plain}{RESET}\n')
        except:
            print(f'Something went wrong while computing.\nPrediction: {RED}{prediction[0]}{RESET} ({YELLOW}{cipher_labels[prediction[0]]}{RESET})\n')
            append_data(f'Something went wrong with the model\'s prediction.\nPrediction was:\n{prediction}({cipher_labels[prediction[0]]})\n')