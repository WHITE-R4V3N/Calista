# Author:       Emma Gillespie
# Date:         2025/02/18
# Description:  Python file that takes in a1z26 and outputs the decoded text.

import string
import sys

letter_array = string.ascii_lowercase

if __name__ == '__main__':
    d_str = ''

    for i in sys.argv[1].split(' '):                # Space should always be the delimeter for a1z26 encodings
        d_str += f'{letter_array[int(i)-1]} '

    print(d_str)                                    # Returns the decoded string by printing to the screen.