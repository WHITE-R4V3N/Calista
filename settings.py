# Author:       Emma Gillespie
# Date:         2024-03-25
# Description:  The settings for the AI model. Mainly colours for colorama

#----------------
#    IMPORTS    |
#----------------

from colorama import Fore

#----------------------
#   GLOBAL SETTINGS   |
#----------------------

CAL_COL = Fore.LIGHTGREEN_EX
RESET = Fore.RESET
GREY = Fore.LIGHTBLACK_EX
YELLOW = Fore.YELLOW
RED = Fore.RED
USER = Fore.CYAN
GREEN = Fore.GREEN

#-----------------------
#   GLOBAL FUNCTIONS   |
#-----------------------

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

    #------------------------------------
    #   Progress usage in other parts   |
    #------------------------------------

    # printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # printProgressBar(e + 1, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)