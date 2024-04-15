# Author:       Emma Gillespie
# Date:         2024-03-28
# Description:  A script that will do a directory search using a wordlist file.

#----------------
#    IMPORTS    |
#----------------

import requests

def find_webpages(base_url, wordlist_file):
  found_pages = []
  with open(wordlist_file, 'r') as f:
    for word in f:
      word = word.strip()  # Remove trailing newline
      url = f"{base_url}/{word}"
      try:
        response = requests.get(url)
        if response.status_code == 200:
          found_pages.append(url)
          print(f"Found webpage: {url}")
      except requests.exceptions.RequestException as e:
        #print(f"Error checking {url}: {e}")
        pass