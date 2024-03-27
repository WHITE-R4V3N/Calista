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
USER = Fore.CYAN

#-----------------------
#   GLOBAL FUNCTIONS   |
#-----------------------

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
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