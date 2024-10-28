# Author:       Emma Gillespie
# Date:         2024-10-27
# Description:  This file contains the parser for the json data. It will parse
#               the training data into np.arrays for the AI to analyze and process.

#----------------
#    IMPORTS    |
#----------------
import json
import numpy as np

# Load the file training_data.json and parse the json data
json_data = json.loads(open("data/training_data.json", 'r').read())

# Function to convert text to numerical data
def text_to_numbers(text):
    return [ord(char.upper()) - 64 for char in text if char.isalpha()]

def numbers_to_text(numbers):
    text = ''.join(chr(num + 64) for num in numbers if 1 <= num <= 26)
    return text

# Function to normalize the data
def normalize_array(arr, max_value=26):
    return arr / max_value
def denormalize_array(arr, max_value = 26):
    return np.round(arr * max_value).astype(int)

def denormalize_prediction(normalized_output):
    denormalize_output = denormalize_array(normalized_output, max_value = 26)
    plaintext = numbers_to_text(denormalize_output[0])

    return plaintext

# Lists for features and labels
cipher_texts = []
keys = []
plain_texts = []
hints = []
algorithms = []

# Extract the features from the JSON data
for entry in json_data["cryptography"]:
    cipher = entry["ciphertext"]
    plain = entry['plaintext']
    key = entry.get('key', 0)           # Default key if none are provided set to 0
    hint = entry['hint']
    algorithm = entry['algorithm']

    cipher_nums = text_to_numbers(cipher)
    plain_nums = text_to_numbers(plain)
    hint_nums = text_to_numbers(hint)
    algorithm_nums = text_to_numbers(algorithm)

    cipher_texts.append(cipher_nums)
    plain_texts.append(plain_nums)
    hints.append(hint_nums)
    keys.append(key)
    algorithms.append(algorithm_nums)

# Pad sequence for uniform shape
max_length = max(len(seq) for seq in cipher_texts) # cipher_texts + plain_texts + hints
cipher_texts = [seq + [0] * (max_length - len(seq)) for seq in cipher_texts]
plain_texts = [seq + [0] * (max_length - len(seq)) for seq in plain_texts]
#hints = [seq + [0] * (max_length - len(seq)) for seq in hints]
algorithms = [seq + [0] * (max_length - len(seq)) for seq in algorithms]

# Convert lists into np.arrays
x_cipher = np.array(cipher_texts)
#x_hints = np.array(hints)
#X = np.hstack([x_hints, x_cipher])
X = x_cipher

y_plain = np.array(plain_texts)
y_algorithm = np.array(algorithms)
#y = np.hstack([y_plain, y_algorithm])
y = y_plain


print(f'Cipher texts and hints (X): \n{X} \nLength: {len(X)} | Length of 1 item: {len(X[0])}')
print(f'\nPlaintexts and algorithms (y): \n{y} \nLength: {len(y)} | Length of 1 item: {len(y[0])}')

# Normalize the X and y arrays
X = normalize_array(X, max_value=26)
y = normalize_array(y, max_value=26)

print(f'\n\nNormalized Cipher texts and hints (X): \n{X} \nLength: {len(X)} | Length of 1 item: {len(X[0])}')
print(f'\nNormalized Plaintexts and algorithm (y): \n{y} \nLength: {len(y)} | Length of 1 item: {len(y[0])}')