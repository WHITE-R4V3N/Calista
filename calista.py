# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  What is and will be considered the front-end of the AI until a website
#               or an app is made rather than the command line. This file is only for
#               dealing with user input and output (mmsg from calista the AI)

#### IMPORTS ####

from settings import *
from model import *

#### CODE ####

logo = f'''

\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{GREY}-----------------------------     
\t\t\t{YELLOW} Prompt     Break     Secure
{RESET}
'''

print(f'{logo}\n\t{CAL_COL}Calista{RESET}> Hello my name is Calista. What would you like me to do?\n')

# This is where the AI will get user input. It will be a loop until ctrl + c or input quit
while True:
    usr_prompt = input(f'\t{USER}User{RESET}> ')

    if usr_prompt.lower() == 'quit':
        print('\n')
        exit(-1)