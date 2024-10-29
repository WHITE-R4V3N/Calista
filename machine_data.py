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
        self.ip = ip        # A unique value for each machine (also what gets checked when users enter a new IP or enters a prompt with an IP)
        self.ports = []
        self.links = []

    def update_ports(self, ports):
        for port in ports:
            self.ports.append(port)

    def update_links(self, links):
        for link in links:
            self.links.append(link)

class MachineManager():
    def __init__(self):
        self.machines = []

    def view_machine(self):
        for machine in self.machines:
            print(f"Machine IP: {machine.ip}\nOther Info:\n{machine.ports}\n{machine.links}")

    def add_machine(self, machine):
        self.machines.append(machine)