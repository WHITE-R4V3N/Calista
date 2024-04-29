# Author:
# Date:         2024-04-27
# Description:  A python program that will parse a website and turn it into readable plain text. Will also grab all
#               ID's and if the item in a textbox, input etc.

#----------
# IMPORTS |
#----------
import re

website_output = '../output.html'
reg_exp = r'<.+?>'  # Will regex for all html tags (ie. <!DOCTYPYE HTML>, <body>, </head>, etc.)
file_data = ''

file = open(website_output, 'r')
file_data = file.read()

html_tags = re.findall(reg_exp, file_data) # Find the tags and store them in a list

print(f'HTML tags that were found:\n{html_tags}\n')

for tag in html_tags:
    file_data = file_data.replace(tag, '').strip() # Replace the tags found with nothing (delete them). Remove white space

print(f'New file data: \n{file_data}\n')