import requests

def get_challeneges(url):
    web_page = requests.get(url).text

    return web_page