# Author:
# Date:
# Description:  A file that holds general functions that can be used.

#----------------
#    IMPORTS    |
#----------------
import pickle
import hashlib
import datetime
import shlex
import subprocess

from colorama import Fore

#--------------------
#   Global colours  |
#--------------------

CAL_COL = Fore.LIGHTGREEN_EX
RESET = Fore.RESET
GREY = Fore.LIGHTBLACK_EX
YELLOW = Fore.YELLOW
RED = Fore.RED
USER = Fore.CYAN
GREEN = Fore.GREEN

#----------------------------------------
#   Function to print a progress bar    |
#----------------------------------------

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} {GREEN}|{bar}|{RESET} {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

#--------------------------------------------------------------------
#   Loading and Saving the model using pickle. Verify with hash.    |
#--------------------------------------------------------------------
# * NOTE * save_model and load_model will need to be refactored and optimized.
# This is not secure in any shape or form.

class ObjectLoadOrSave:
    def save_object(self, obj, filename):
        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(obj, f)

        hash_func = hashlib.new('sha256')

        with open(f'{filename}.pkl', 'rb') as f:
            while  chunk := f.read(1024):
                hash_func.update(chunk)

        print(f'{GREY}[common-save-object-hash]{RESET} {filename}\n{hash_func.hexdigest()}\n')

    def load_object(self, filename):
        with open(f'{filename}.pkl', 'rb') as f:
            obj = pickle.load(f)
            
        hash_func = hashlib.new('sha256')

        with open(f'{filename}.pkl', 'rb') as f:
            while  chunk := f.read(1024):
                hash_func.update(chunk)

        print(f'{GREY}[common-load-object-hash]{RESET} {filename}\n{hash_func.hexdigest()}\n')

        return obj
    
#----------------------------
#   Dealing with the logs   |
#----------------------------
class Log:
    def __init__(self):
        # Open the file or create the file based on yyyy-mm-dd-
        self.filename = str(datetime.datetime.now()).split(' ')[0]
        self.file = ''

    def open_logs(self):
        self.file = open(f'logs/{self.filename}', 'a')

    def write_logs(self, text):
        self.file.write(f'{text} | {str(datetime.datetime.now())}\n')

    def close_logs(self):
        self.file.close()

class PreviewExecution:
    def __init__(self):
        self.key_words = ['a1z26', 'rot13', 'base64']

    def preview_execution(self, command_str):
        if command_str not in self.key_words:
            try:
                parts = shlex.split(command_str)
                print(f'\n{YELLOW}[-]{RESET} Preview (sandboxed): {command_str}\n')
                result = subprocess.run(parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f'{result.stdout.decode()}\n')
            except Exception as e:
                print(f'{RED}[x]{RESET} Execution Error: {e}\n')
            
            return True
        else:
            return False
        
class TokenToUserPrompt:
    def __init__(self):
        pass