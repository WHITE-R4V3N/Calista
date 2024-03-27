# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  What is and will be considered the front-end of the AI until a website
#               or an app is made rather than the command line. This file is only for
#               dealing with user input and output (mmsg from calista the AI)

#----------------
#    IMPORTS    |
#----------------

from settings import *
from model import *

#----------------
#    I/O Loop   |
#----------------

logo = f'''
\n\n\n\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------
{RESET}
'''

print(f'{logo}\n{CAL_COL}Calista{RESET}> Hello my name is Calista. I am an AI that is capable of hacking.\n\t Enter a prompt to get started.\n')

# This is where the AI will get user input. It will be a loop until ctrl + c or input quit
while True:
    usr_prompt = input(f'{USER}User{RESET}> ')

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):
        print('\n')
        exit(-1)
    else:
        print(f'\n{CAL_COL}Calista{RESET}> Let me see what I can do based on your input.\n')