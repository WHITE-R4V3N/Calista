# Author:       Emma Gillespie
# Date:         2024-03-26
# Description:  A script that will scrape a website to pull the important information needed from it for the parser/tokenizer.

#----------------
#    IMPORTS    |
#----------------

import requests
from bs4 import BeautifulSoup
import os
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By

def scrape_and_save(url, output_file):
    # Send a GET request to the URL
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        # Save the HTML content to a local file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(str(soup))
        print(f"Website scraped and saved to {output_file}\n")
    else:
        print(f"Failed to retrieve website content. Status code: {response.status_code}\n")

def open_local_html_file(html_file):
    # Open the local HTML file in the default web browser
    webbrowser.open(f'file://{os.path.abspath(html_file)}')

# What the webscrapper will be switched to potentially.
# This will also be used to submit flags to the webpage for the CTF
# The find element by ID will be done systematically by the program so the user doesn't
# need to enter any data into the program
def interact_with_webpage():
    driver = webdriver.Chrome() # Define the web browsers

    driver.get('') # Get the webpage and website.

    element = driver.find_element(By.ID, "id") # Gets the element by the ID

    element.send_keys("Arrays") # Data to be sent to the webpages