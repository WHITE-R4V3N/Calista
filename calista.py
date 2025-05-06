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

from model import *

# -------------------------------
# Global Variables
# -------------------------------
version = 'v 3.0.0'
logo = f'''
\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------{RESET} {BLUE}{version}{RESET}
\t\t\t\t\t   By: Emma Gillespie

{RED}[DISCLAIMER]{RESET} Calista is a capstone project and only to be used for ethical purposes!
'''

max_seq_len = 8

# -------------------------------
# Objects and tokenizer setup
# -------------------------------
tokenizer = Tokenizer() # Has a function to create and build the vocab for the model.

example_dataset = [
    ("list files", "ls"),
    ("change directory to home", "cd ~"),
    ("make a directory called test", "mkdir test"),
    ("remove file data.txt", "rm data.txt"),
    ("show current directory", "pwd"),
    ("display contents of file", "cat file.txt"),
    ("search for text in file", "grep 'text' file.txt"),
    ("copy file a to b", "cp a b"),
    ("move file x to folder y", "mv x y"),
    ("create empty file named log", "touch log")
]

dataset, X, y = tokenizer.parse_datasets(example_dataset)
tokenizer.build_vocab(dataset)

model = Transformer(d_model=32, num_heads=2, d_ff=64, src_vocab_size=len(tokenizer.word2idx)+1, tgt_vocab_size=len(tokenizer.word2idx)+1, max_seq_len=max_seq_len)
#                                                       len of vocab                                len of vocab                      len of commands
print(f"[model] {model}")

# -------------------------------
# Setup dataset and model
# -------------------------------
temp_X = []
for i in X:
    temp_X.append(tokenizer.encode(i, max_seq_len))
    #                                max length of sequence 8
X = np.array(temp_X)

temp_y = []
for j in y:
    temp_y.append(tokenizer.encode(j, max_seq_len))
    #                                max length of sequence 8
y = np.array(temp_y)

print(f'[X parsed] \n{X}')
print(f'[y parsed] \n{y}')

train(model, X, y, epochs=5)

# -------------------------------
# Test the model on new input
# -------------------------------
test_input = "show current directory"
test_ids = np.array(tokenizer.encode(test_input, max_seq_len))
dummy_tgt = np.array([[1] [0] * (max_seq_len - 1)])
#                               max sequence length 8

pred_logits = model.forward(test_ids, dummy_tgt)
pred_command = np.argmax(pred_logits[0], axis=-1)
print(f"Generated Command: {tokenizer.decode(pred_command)}")

print(logo)