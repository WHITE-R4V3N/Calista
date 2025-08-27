#   Author:         Emma Gillespie
#   Date:           2025-08-03
#   Description:    This file is responsible for scanning the ports on a target machine
#                   and saving the data as a json file. This data may be saved in a file
#                   or just as a class object that can be called or edited.

#----------------
#    IMPORTS    |
#----------------
import socket
import json
import argparse
import subprocess
import requests

from platform import system
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore
#from common import *

CAL_COL = Fore.LIGHTGREEN_EX
RESET = Fore.RESET
GREY = Fore.LIGHTBLACK_EX
YELLOW = Fore.YELLOW
RED = Fore.RED
USER = Fore.CYAN
GREEN = Fore.GREEN

COMMON_PORTS = {
    21: 'FTP',
    22: 'SSH',
    23: 'Telnet',
    25: 'SMTP',
    53: 'DNS',
    80: 'HTTP',
    110: 'POP3',
    143: 'IMAP',
    443: 'HTTPS',
    3306: 'MySQL',
    3389: 'RDP',
    8080: 'HTTP-Alt'
}

VULNERABLE_BANNERS = {
    'OpenSSH_7.2': 'CVE-2016-0777: Infromation disclosure vulnerability in OpenSSH' # Better if this was a dict or list not manually typed
}

def get_banner(ip, port, timeout):
    try:
        with socket.socket() as s:
            s.settimeout(timeout)
            s.connect((ip, port))
            return s.recv(1024).decode(errors="ignore").strip()
    except:
        return ''
    
def scan_port(ip, port, timeout):
    result = {
        'port': port,
        'status': 'closed',
        'service': COMMON_PORTS.get(port, 'unknown'),
        'banner': ''
    }

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)

            if s.connect_ex((ip, port)) == 0:
                result['status'] = 'open'
                result['banner'] = get_banner(ip, port, timeout)

                vuln_info = check_vulnerabilities(result['banner'])

                if vuln_info:
                    result['vulnerability'] = vuln_info
                else:
                    result['vulnerability'] = ''
    except:
        pass

    return result

def scan_target(host, start_port, end_port, timeout, max_threads):
    print(f'\nStarting scan on {host} from port {start_port} to {end_port}')
    print(f'Scan Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n')

    try:
        ip = socket.gethostbyname(host)
    except:
        print(f'{RED}[x]{RESET} Unable to resolve host: {host}\n')
        return
    
    results = []

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(scan_port, ip, port, timeout)
            for port in range(start_port, end_port+1)
        ]

        for future in as_completed(futures):
            res = future.result()
            results.append(res)

            if res['status'] == 'open':
                print(f'{GREEN}[+]{RESET} Port {res['port']} OPEN ({res['service']})')

                if res['banner']:
                    print(f'\t| Banner: {res['banner']}')
                print()

    os_guess = guess_os_ttl(ip)
    scan_report = {
        'target': host,
        'ip': ip,
        'os': os_guess,
        'start_port': start_port,
        'end_port': end_port,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }

    output_file = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    if output_file:
        with open(f'../recon/{output_file}', 'w') as f:
            json.dump(scan_report, f, indent=4)
        print(f'{YELLOW}[-]{RESET} Scan results saved to recon/{output_file}.')

    print(f'\n Scan completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')

def guess_os_ttl(ip):
    try:
        if system().lower() == 'windows':
            cmd = ['ping', '-n', '1', ip]
        else:
            cmd = ['ping', '-c', 1, ip]

        output = subprocess.check_output(cmd).decode(errors='ignore')

        if 'ttl=' in output.lower():
            ttl_value = int(output.lower().split('ttl=')[1].split()[0])

            if ttl_value >= 128:
                return 'Windows (TTL ~128)'
            elif ttl_value >= 64:
                return 'Linux/Unix (TTL ~64)'
            elif ttl_value >= 255:
                return 'Cisco/Networking Device (TTL ~255)'
            else:
                return f'Unknown OS (TTL: {ttl_value})'
    except Exception as e:
        return f'OS fingerprinting failed: {e}\n'
    return 'Unknown'

def check_vulnerabilities(banner):
    # We need to analyze and the split the data, and parse to get the specific data needed. # Do this at home tonight or tomorrow morning

    # Banner from a previous test
    # SSH-2.0-OpenSSH_9.2p1 Debian-2+deb12u3
    banner = 'SSH-2.0-OpenSSH_9.2p1 Debian-2+deb12u3'

    cpe = banner.replace('-', ':').replace(' ', ':')

    url = 'https://services.nvd.nist.gov/rest/json/cves/2.0'
    params = {"cpeName": cpe}

    response = requests.get(url, params=params)
    data = response.json()

    for vuln in data.get('vulnerabilities', []):
        cve = vuln['cve']['id']
        desc = vuln['cve']['descriptions'][0]['value']
    print(f'{cve}: {desc}\n')
    
    return ''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nmap-like port scanner in Python")
    parser.add_argument('host', help="Target host to scan (IP or DOMAIN)")
    parser.add_argument('-s', '--start', type=int, default=1, help='Starting port number')
    parser.add_argument('-e', '--end', type=int, default=1024, help='End port number')
    parser.add_argument('-t', '--timeout', type=float, default=0.3, help='Sets the timeout for connection')
    parser.add_argument('-m', '--max_threads', type=int, default=10, help='Max number of threads to create')
    args = parser.parse_args()

    scan_target(args.host, args.start, args.end, args.timeout, args.max_threads)   # Output file name can be date/time
                                                                                                # Instead of determined by user