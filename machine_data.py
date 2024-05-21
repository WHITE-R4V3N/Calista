# Author:       Emma Gillespie
# Date:         2024-05-15
# Description:  This will create a class object for each machine the user wishes to attack.
#               This will be based on each new ip address the user enters in the prompts.

#----------------
#    IMPORTS    |
#----------------

from settings import *

class Machine():
    def __init__(self, ip):
        self.ip = ip        # A unique value for each machine (also what gets checked when users enter a new IP)
        self.ports = []
        self.links = []

    def update_ports(self, port):
        self.ports.append(port)

    def update_links(self, link):
        self.links.append(link)