#   Author:         Emma Gillespie
#   Date:           2025-07-01
#   Description:    This script will decode the ciphers being used in the dataset. If there is a
#                   cipher it does not understand then it will flag it.

#--------------
#   Imports   |
#--------------
import base64

from common import *

def decode_base64(text):
    print(f'\n{YELLOW}[-]{RESET} Attempting to decode...\n')
    try:
        decoded_text = base64.b64decode(text)
        print(f'{GREY}[+]{RESET} {decoded_text}\n')
    except:
        print(f'{RED}[x]{RESET} Failed to decode using {YELLOW}base64{RESET}.\n')

    return decoded_text if decoded_text else 'Failed to decode.'

def decode_a1z26(text):
    print(f'\n{YELLOW}[-]{RESET} Attempting to decode...\n')

    try:
        text = text.split(' ')
        decoded_text = ''

        for i in text:
            if i not in ['{', '}']:
                decoded_text += chr(int(i)+96)
            else:
                decoded_text += i

        print(f'{GREEN}[+]{RESET} {decoded_text}\n')
    except:
        print(f'{RED}[x]{RESET} Failed to decode using {YELLOW}a1z26{RESET}.\n')
    
    return decoded_text if decoded_text else 'Failed to decode.'

def decode_rot13():
    pass