import requests
from bs4 import BeautifulSoup as bs
import re
import urllib.request

"""
This scripts allows to scrap several watches images from Chrono24 website in order to create
a dataset for later model trainnings.

Sometimes, searching url can be changed.
"""


brand = 'brand'
model = 'model'

pic_links = []
count = 0


# Browser headers, required to access Chrono24
headers = {'User-Agent':
           'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko)'
           'Chrome/50.0.2661.102 Safari/537.36'}


# Getting all the images links through 30 pages
for i in range(1, 30):
    url = f'https://www.chrono24.es/{brand}/index-{i}.htm?query={model}'
    html = requests.get(url).content
    soup = bs(html, 'lxml')
    soup = soup.find_all('div', {'class': 'article-image-container'})
    pic_list = [i[8:] for i in re.findall('srcset.*.jpg', str(soup))]
    pic_links = pic_links + pic_list
    count += 1
    # print(pic_list)

# Fetching every image related to the list above
count = 1
opener = urllib.request.URLopener()
opener.addheader(headers)

for i in pic_links:
    print(i)

    # You can set the image name
    img_name = 'Radiomir' + str(count)
    print(img_name)

    # Set the path to save the images
    filename, headers = opener.retrieve(i, f'./data/imgs/Panerai_Models/Radiomir/{img_name}.jpeg')
    print(f'{count}')

    count += 1
