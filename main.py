#   Author:         Emma Gillespie
#   Date:           2025-06-14
#   Description:    This is a refactored and improved version of the simple_feedforward neural network
#                   that is being developed and used for my capstone project. This is the main file to
#                   controll the user input and model interaction. SLM (Small Language Model)

#----------------
#    IMPORTS    |
#----------------
import numpy as np
import json
import os

from model import *
from ctokenizer import *
from common import *

from scripts.cryptography import *
from scripts.recon import *

#--------------------------------------------------------------
#   Global variables, Tokenizer, Logging, Loading Datasets    |
#--------------------------------------------------------------
model_type = 'ff'   # ff (feed forward) or nlp (transformer)
debug_mode = True
execution = False

input_size = 16     # Used for the feed forward network
output_size = 8    # Variable so may need to change this
loss = 0

obj_manager = ObjectLoadOrSave()
preview_command = PreviewExecution()
logs = Log()
logs.open_logs()
logs.write_logs(f'[main-model_type] {model_type}')

tokenizer = Tokenizer(mode='word')

############################

with open('../ai_datasets/live_dataset.json', 'r') as f:
    dataset = json.load(f)

############################

'''
with open('../ai_datasets/testing_dataset.json', 'r') as f:
    dataset = json.load(f)

with open('../ai_datasets/command_dataset.json', 'r') as f:
    dataset = json.load(f)

with open('../ai_datasets/cryptography_dataset.json', 'r') as f:
    dataset += json.load(f)

with open('../ai_datasets/recon_dataset.json', 'r') as f:
    dataset += json.load(f)

with open('../ai_datasets/web_dataset.json', 'r') as f:
    dataset += json.load(f)
'''

x_text = [item['prompt'] for item in dataset]
y_text = [item['response'] for item in dataset]
combined_data = x_text + y_text

# Padd the output so it is even and the same over all of the data.
# This will help with the output being smaller than its supposed to.

tokenizer.build_vocab(combined_data)
vocab_size = len(tokenizer.word2idx)

if model_type == 'ff':
    encoded_x = [tokenizer.encode(text, input_size) for text in x_text]
    encoded_y = [tokenizer.encode(text, output_size) for text in y_text]

    model = FeedForwardNetwork(input_size, 512, 512, output_size, debug=debug_mode)

    one_hot_y = [[int(i == item) for i in sorted(encoded_y)] for item in encoded_y]
    max_x = max(max(encoded_x))
    normalized_x = [[val / max_x for val in row] for row in encoded_x]

    # Then train the model
    if debug_mode == True:
        loss = model.train(np.array(normalized_x), np.array(one_hot_y), 20000 if (model_type == 'ff') else 500 if (model_type == 'nlp') else 0)

elif model_type == 'nlp':
    encoded_x = [tokenizer.encode(text, input_size) for text in x_text]
    encoded_y = [tokenizer.encode(text, input_size) for text in x_text]

    model = Transformer(input_size=vocab_size, output_size=vocab_size, learning_rate=0.001, debug=debug_mode)

    # Use Class Indicies rather than one-hot encoding (for now one-hot encoding)
    one_hot_y = []
    for y in encoded_y:
        onehot = np.zeros(vocab_size)
        for t in y:
            onehot[t] = 1
        one_hot_y.append(onehot)

    # Then train the model
    if debug_mode == True:
        loss = model.train(np.array(encoded_x), np.array(one_hot_y), epochs=250)

if loss:
    print(f'\n{GREY}[main-loss]{RESET}\n{loss:.4f}\n')
    logs.write_logs(f'[main-loss] {loss:.4f}')

#--------------------------------
#   Save and/or Load Objects    |
#--------------------------------

if debug_mode == True:
    obj_manager.save_object(model, f'model_{model_type}')
    obj_manager.save_object(tokenizer, f'tokenizer_{model_type}')

'''
else:
    obj_manager.load_object(f'model_{model_type}')
    obj_manager.load_object(f'tokenizer_{model_type}')
'''
    
#----------------------------
#   User interaction loop   |
#----------------------------
logo = f'''
\n\n\n\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------{RESET} v 3.2.1
\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Capstone project and only to be used for ethical purposes!
'''

print(f'{logo}')

while True:
    execution = False
    user_prompt = input(f'{USER}User{RESET}> ')

    if user_prompt == '1':
        print(f'Preset Promot: {YELLOW}scan machine 192.168.1.72{RESET}')
        user_prompt = 'scan machine 192.168.1.72'
    elif user_prompt == '2':
        print(f'Preset Promot: {YELLOW}list all files{RESET}')
        user_prompt = 'list all files'
    elif user_prompt == '3':
        print(f'Preset Promot: {YELLOW}generate ssh key{RESET}')
        user_prompt = 'generate ssh key'
    elif user_prompt == '4':
        cipher = '16 18 5 6 9 24 { 20 8 5 14 21 13 2 5 18 19 13 1 19 15 14 }'
        print(f'Preset Promot: {YELLOW}{cipher}{RESET}')
        user_prompt = cipher
    elif user_prompt == '5':
        print(f'Preset Promot: {YELLOW}cHJlZml4e2Jhc2U2NF9kZWNyeXB0aW9uX3N1Y2Nlc3N9{RESET}')
        user_prompt = 'cHJlZml4e2Jhc2U2NF9kZWNyeXB0aW9uX3N1Y2Nlc3N9'
    elif user_prompt == '6':
        print(f'Preset Promot: {YELLOW}Enumerate available directories on a website{RESET}')
        user_prompt = 'Enumerate available directories on a website'

    for token, pattern in tokenizer.custom_tokens.items():
        tokenizer.user_specials[token] = re.findall(pattern, user_prompt)
        user_prompt = re.sub(pattern, token, user_prompt)
    logs.write_logs(f'[user] {user_prompt}')

    if user_prompt.lower() in ('q', 'quit'):
        logs.write_logs('[main-quit] Shutting down the AI.')
        logs.close_logs()
        print()
        break

    if user_prompt.lower() in ('cls', 'clear'):
        os.system('clear')
        continue

    if model_type == 'ff':
        user_encoded = tokenizer.encode(user_prompt, input_size)
        logs.write_logs(f'[main-user_encoded] {user_encoded}')
        #print(f'[user encoded]\n{user_encoded}\n')
        #print(f'[main-user_specials]\n{tokenizer.user_specials}\n')

        user_normalized = [val / max_x for val in user_encoded]
        logs.write_logs(f'[main-user_normalized] {user_normalized}')

        model_prediction = model.forward(user_normalized)
        confidence_score = model.compute_confidence_score(model_prediction)

        #print(f'[model prediction]\n{model_prediction}\n')#

        #print(f'{GREY}\n[main-prediction]{RESET}\n{model_prediction}\n')
        print(f'\nConfidence Score = {GREEN if confidence_score*100 > 50 else YELLOW }{confidence_score*100:.2f}%{RESET}\n')
        logs.write_logs(f'[main-prediction] {model_prediction}')
        logs.write_logs(f'[main-confidence] Confidence Score = {confidence_score*100:.2f}%')

        # We need to take the prediction and convert it back into what the AI needs to do
        rounded_prediction = [[round(i) for i in vec] for vec in model_prediction]
        one_hot_prediction = [sorted(encoded_y)[vec.index(1)] for vec in rounded_prediction]

        #print(f'[rounded_prediction]\n{rounded_prediction}\n')#
        #print(f'[one_hot_prediction]\n{one_hot_prediction}\n')#

        logs.write_logs(f'[main-one_hot_prediction] {one_hot_prediction}')
        logs.write_logs(f'[main-rounded_prediction] {rounded_prediction}')

        #print(f'{GREY}[main-rounded_prediction]{RESET}\n{rounded_prediction}\n')
        #print(f'{GREY}[main-one_hot_prediction]{RESET}\n{one_hot_prediction}\n')

        decoded_prediction = tokenizer.decode(one_hot_prediction[0])

        for special_token in tokenizer.user_specials:
            if special_token in decoded_prediction:
                special_data = tokenizer.user_specials[special_token]
                try:
                    decoded_prediction = decoded_prediction.replace(special_token, special_data[0])
                except:
                    pass
                
        print(f'Command: {GREEN}{decoded_prediction}{RESET}\n')
        logs.write_logs(f'[main-decoded-prediction] {decoded_prediction}')

        choice = input(f'{CAL_COL}Calista{RESET}> Whould you like to run the predicted command (y/n)? ')
        
        if choice.lower() in ('y', 'yes'):
            execution = preview_command.preview_execution(decoded_prediction)
        else:
            print()

        if not execution and choice.lower() in ('y', 'yes'):
            if decoded_prediction == 'a1z26':
                decoded_text = decode_a1z26(user_prompt)
            elif decoded_prediction == 'rot13':
                decoded_text = decode_rot13()
            elif decoded_prediction == 'base64':
                decoded_text = decode_base64(user_prompt)
            else:
                decoded_text = 'Could not find cipher used.'
            
            logs.write_logs(f'Decoded text: {decoded_text}')
        
    elif model_type == 'nlp':
        pass
