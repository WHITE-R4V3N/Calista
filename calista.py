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

print(logo)