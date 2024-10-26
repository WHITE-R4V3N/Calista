# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  What is and will be considered the front-end of the AI until a website
#               or an app is made rather than the command line. This file is only for
#               dealing with user input and output (mmsg from calista the AI)

#----------------
#    IMPORTS    |
#----------------

import codecs

from settings import *
from model import *
from tokenizer import *
from machine_data import *

from scripts.scan_ports import *
from scripts.network_map import *
from scripts.scrape_website import *
from scripts.dirb import *

machines = MachineManager()
current_machine = ''

# NN Model's
# nn_models = [calista_model, recon_model, ...]
calista_model = ''              # Processing input from the user and deciding what model(s) to use
recon_model = ''                # Used for processing input related to recon
cryptography_model = ''         # Used for input related to cryptographic analysis
reverse_engineering_model = ''  # Used for prompts related to reverse engineering
forensics_model = ''            # Used for input related to forensics
general_skills_model = ''       # Used for inputs related to general skills in cyber security
binary_exploit_model = ''       # Used for inputs related to binary exploitation
web_exploit_model = ''          # Used for input related to web exploitation

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
\t\t\t{YELLOW}-----------------------------{RESET}\tv 0.0.9
'''

print(f'{logo}\n{CAL_COL}Calista{RESET}> Hello my name is Calista. I am an AI that is capable of competing\n\t in a cyber CTF.\n\n\t Enter questions from the CTF and I will do the rest.\n')

# This is where the AI will get user input. It will loop until the user inputs quit.
while True:
    usr_prompt = input(f'{USER}User{RESET}> ')              # Get user input
    usr_bin_array = create_bin_array(corpus, usr_prompt)    # Create a binary array using the corpus and user input.

    if (usr_prompt.lower() == 'quit') or (usr_prompt.lower() == 'q'):
        print('\n')
        exit(-1)
    else:
        data = np.array(usr_bin_array)

        tasks = np.array([1, 1, 1, 1, 1])
        prediction = model.predict(data)
        
        # Create the machine object or load data if machine with IP already exists during session.
        try:
            ip_address = re.findall(ip_pattern, usr_prompt)[0] # Finds an IP address in the user input.
            
            if (not machines): # if no machines exist then create the first machine
                machines.add_machine(Machine(ip_address))
            elif (ip_address not in [machine.ip for machine in machines.machines]):# If IP address not found with in the machines then create
                machines.add_machine(Machine(ip_address))    # the machine object with ip equal to the user input IP
            #else:
            for machine in machines.machines:
                if (ip_address == machine.ip):
                    current_machine = machine
        except:
            pass

        #-------------------------------------------------------------
        #    Program what each prediction should do individually.    |
        #-------------------------------------------------------------
        for pred in prediction.round():
            if pred[0] == tasks[0]:
                #----------------------------------------------------------------------------
                #   Will run the port_scan.py script. This will find ports between 1-1000   |
                #----------------------------------------------------------------------------
                ip_address = re.findall(ip_pattern, usr_prompt)[0]
                print(f'\n{CAL_COL}Calista{RESET}> Scanning {ip_address} for open ports.')
                
                ports = [range(1, 1000)]
                for p in ports:
                    open_ports = scan_target(ip_address, p) # The AI should be able to pull this information from the prompt

                current_machine.update_ports(open_ports)
        
        #machines.view_machine() Used to see whats happening with the machine