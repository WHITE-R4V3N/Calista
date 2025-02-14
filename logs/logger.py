#   Author:         Emma Gillespie
#   Date:           2024-11-14
#   Description:    File containing functions for creating new logs and log entry's for
#                   the AI and so we can see what the AI is doing.

import datetime

# Create the log file
x = str(datetime.datetime.now()).split(' ')

def create_file():
    log = open(f"./logs/{x[0]}.txt", "a")
    log.write(f'*CALISTA AI STARTED*\n\nFile edited at {x[1]} of {x[0]}\n\n')
    log.close

def append_data(data):
    log = open(f'./logs/{x[0]}.txt', 'a')
    log.write(f'{'-'*40}\n{data}\n')
    log.close