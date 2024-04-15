# Author:       Emma Gillespie
# Date:         2024-03-26
# Description:  A script that will map a subnet and give all device IP addresses

#----------------
#    IMPORTS    |
#----------------
import nmap

def scan_network(subnet):
    nm = nmap.PortScanner()
    nm.scan(hosts=subnet, arguments='-sn')

    devices = []
    for host in nm.all_hosts():
        if 'mac' in nm[host]['addresses']:
            mac_address = nm[host]['addresses']['mac']
        else:
            mac_address = "Unknown"
        devices.append({
            'ip': host,
            'mac': mac_address,
            'vendor': nm[host].get('vendor', "Unknown"),
            'hostname': nm[host].get('hostname', "Unknown")
        })
    return devices