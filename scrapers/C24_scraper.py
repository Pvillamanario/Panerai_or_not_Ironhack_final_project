import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
from datetime import datetime
import urllib.request # save img by url
import os # imgs directory creation
import html5lib


def c24_imgs_scraper(model):

    """Scraps pictures from chrono24.com to improve model training.
        Accepts as argument a brand or a model"""

    # Needed to open this web
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

    pic_links = []
    count = 0
    for i in range(1, 29):
        url = f'https://www.chrono24.es/panerai/index-{i}.htm?query=radiomir'
        html = requests.get(url).content
        soup = bs(html, 'lxml')
        soup = soup.find_all('div', {'class': 'article-image-container'})
        pic_list = [i[8:] for i in re.findall('srcset.*.jpg', str(soup))]

        count += 1

        print(count)

