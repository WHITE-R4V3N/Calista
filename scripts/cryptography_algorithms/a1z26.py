# Author:       Emma Gillespie
# Date:         2025/02/18
# Description:  Python file that takes in a1z26 and outputs the decoded text.

import string
import sys

letter_array = string.ascii_lowercase

def decrypt_a1z26(string):
    d_str = ''

    for i in string:
        d_str += f'{letter_array[int(i)-1]}'

    return d_str