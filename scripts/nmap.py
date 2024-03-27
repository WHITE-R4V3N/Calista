# Author:       Emma Gillespie
# Date:         2024-03-26
# Description:  A script that is a similar fashion to nmap that the AI can use for gathering information

#----------------
#    IMPORTS    |
#----------------

import socket
from concurrent.futures import ThreadPoolExecutor

#---------------------------------------
#   Functions for the scan of device   |
#---------------------------------------

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
    print(f"\nScanning target {target}...")
    print('\nPORT    STATE    SERVICE')
    with ThreadPoolExecutor(max_workers=100) as executor:
        for port in ports:
            executor.submit(scan_port, target, port)

#--------------------------------------------------------
#   Use as a guide for how the AI can use this script   |
#--------------------------------------------------------

#def main():
#    target = input("\nEnter the target IP address or hostname: ")
#    usr_input = input("Enter ports to scan (range, e.g., 1-1000): ").split('-')
#
#    ports = [range(int(usr_input[0]), int(usr_input[1]))]
#
#    for p in ports:
#        scan_target(target, p)
#
#    print('')
#
#if __name__ == "__main__":
#    main()
