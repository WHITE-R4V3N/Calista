# Author:       Emma Gillespie
# Date:         2024-03-26
# Description:  A script that is a similar fashion to nmap that the AI can use for gathering information

#----------------
#    IMPORTS    |
#----------------

import socket
from concurrent.futures import ThreadPoolExecutor

#------------------------------------------------------------
#   Functions for the scan of device and global variables   |
#------------------------------------------------------------

found_services = []

def scan_port(target, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            result = s.connect_ex((target, port))
            if result == 0:
                try:
                    service = socket.getservbyport(port)
                except OSError:
                    service = "Unknown"
                print(f'{port}/tcp  open     {service}')
    except Exception as e:
        pass

def scan_target(target, ports):
    print(f"Scanning target {target}...")
    print('\nPORT    STATE    SERVICE')
    with ThreadPoolExecutor(max_workers=100) as executor:
        for port in ports:
            executor.submit(scan_port, target, port)