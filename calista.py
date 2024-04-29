# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  What is and will be considered the front-end of the AI until a website
#               or an app is made rather than the command line. This file is only for
#               dealing with user input and output (mmsg from calista the AI)

#----------------
#    IMPORTS    |
#----------------

from settings import *
from model import *
from scripts.scan_ports import *
from scripts.network_map import *
from scripts.scrape_website import *
from scripts.dirb import *

#----------------
#    I/O Loop   |
#----------------

logo = f'''
\n\n\n\n
\t\t\t   ___      _ _     _        
\t\t\t  / __\\__ _| (_)___| |_ __ _ 
\t\t\t / /  / _` | | / __| __/ _` |
\t\t\t/ /__| (_| | | \\__ \\ || (_| |
\t\t\t\\____/\\__,_|_|_|___/\\__\\__,_|
\t\t\t{YELLOW}-----------------------------
{RESET}
'''

print(f'{logo}\n{CAL_COL}Calista{RESET}> Hello my name is Calista. I am an AI that is capable of hacking.\n\t Enter a prompt to get started.\n')

# This is where the AI will get user input. It will be a loop until ctrl + c or input quit
while True:
    usr_prompt = input(f'{USER}User{RESET}> ')

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):
        print('\n')
        exit(-1)
    else:
        print(f'\n{CAL_COL}Calista{RESET}> Let me see what I can do based on your input.\n')

        data = np.array([int(i) for i in usr_prompt.split(' ')]) # Temp parsing of data

        # Scan a device, Scan a network, scrape website, dirb
        tasks = np.array([1, 1, 1, 1, 1]) # This can be changed. To a dictionary rather than a list. Maybe....
        prediction = model.predict(data)

        for pred in prediction.round():
            if pred[0] == tasks[0]:
                #-----------------------------------------------------------------------
                #   Where we use the scan_ports.py script based on the AI prediction   |
                #-----------------------------------------------------------------------

                ports = [range(1, 1000)]

                for p in ports:
                    scan_target('192.168.1.72', p) # The AI should be able to pull this information from the prompt

                print('')

            if pred[1] == tasks[1]:
                #--------------------------------------------------------------------
                #   Where we use the network_map.py script based on AI prediction   |
                #--------------------------------------------------------------------

                subnet = "192.168.1.0/24"

                print(f'Scanning the network {subnet}...')

                devices = scan_network(subnet)
                print("\nDevices found:")
                for device in devices:
                    print(f"IP: {device['ip']}, MAC: {device['mac']}, Vendor: {device['vendor']}, Hostname: {device['hostname']}")
                
                print('')
            if pred[2] == tasks[2]:
                #-----------------------------------------------------------------
                #   This is where we will curl the website (for now just open)   |
                #-----------------------------------------------------------------

                url = "http://192.168.1.72/"
                output_file = 'output.html'

                scrape_and_save(url, output_file)
                open_local_html_file(output_file)

            if pred[3] == tasks[3]:
                #----------------------------------------------------------------------------------
                #   This will do a directory search of the website and find all common webpages   |
                #----------------------------------------------------------------------------------

                print('Searching for webpages...')
                print("** Please note:** This script performs basic checks and may not identify all existing webpages. Always respect robots.txt guidelines when scraping websites.\n")
                try:
                    find_webpages('http://192.168.1.72', "scripts/wordlist.txt")
                except:
                    pass
                #find_webpages(base_url, wordlist_file)
                print('')

            if pred[4] == tasks[4]:
                #
                #
                #
                pass
