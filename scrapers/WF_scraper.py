import requests
from bs4 import BeautifulSoup as bs
import re
import pandas as pd
from datetime import datetime
import urllib.request
import os
import shutil

"""
This scripts updates the info about Panerai watches being selled on Watchfinder. It fetches all the watch 
info and images.
"""

# Links list path
df_links_path = '../data/wfinder_panerai_links.csv'


def get_links_panerai_on_sale():
    """
    Retrieves the links of all Panerai watches on sale at watchfinder.es (so prices are in euros)
    and save the list to be used later.
    """

    links = []

    print('Scrapping Panerai watches on sale at watchfinder.co.uk')
    count = 1

    for page in range(1, 140):

        url = f'https://www.watchfinder.co.uk/all-watches?orderby=BestMatch&pageno={page}'
        html = requests.get(url).content
        soup = bs(html, 'lxml')
        soup = soup.find_all('a', {'class': 'prods_name redirect'})

        # Cleaning the link
        for i in re.findall(r'href.*">', str(soup)):
            link = i[6:-10].replace(' ', '')
            link = re.findall('.*sea', link)[0][:-4]

            if 'Panerai' in link:
                # print(link)
                links.append(link)
        count += 1

        print(f'\rScraping web {count} of 130', end="", flush=True)

    print(f'{len(links)} watches retrieved.')

    # Saving the link list to csv
    df_links = pd.DataFrame(links)
    df_links.to_csv(df_links_path, index=False)


def info_and_imgs_panerai_on_sale():

    """
    Itinerates through the link list ang gets the info about the watch on sale and saves de img.
    """

    df_features = pd.DataFrame(columns=['id', 'model', 'pam',
                                        'price', 'retail_price',
                                        'year', 'box', 'papers',
                                        'gender', 'movement', 'case_size',
                                        'case_material', 'bracelet_material',
                                        'dial_type', 'w_resistance',
                                        'date', 'link', 'img_path'])

    df_links = pd.read_csv('../data/wfinder_panerai_links.csv')
    home_url = 'https://www.watchfinder.es'
    count = 1
    print('Link list retrieved.')

    # Deleting old images
    print('Cleaning old images.')
    shutil.rmtree('../data/imgs/WF_panerai', ignore_errors=True)
    os.mkdir('../data/imgs/WF_panerai')

    for i in df_links['0']:

        try:
            # url for each watch
            watch_id = i[-6:]
            watch_url = home_url + i

            # Watch features table
            watch_table = pd.read_html(watch_url)[0].set_index(0)

            # bs soup
            watch_html = requests.get(watch_url).content
            watch_soup = bs(watch_html, 'lxml')

            model = watch_soup.find_all('span', {'class': 'prod_series ellipsis'})
            model = str(model)[36:-8].replace('.', '').strip()

            pam = re.findall('PAM\d{5}', i)
            pam = pam[0]

            price = watch_soup.find_all('span', {'class': ''})[4]
            price = re.findall(r'€.*<', str(price))
            price = ''.join(re.findall(r'\d', str(price)))

            # Some data doesn't always appear
            try:
                retail_price = watch_soup.find_all('div', {'class': 'prod_price-info notranslate'})[0]
                retail_price = ''.join(re.findall(r'\d', str(retail_price)))
            except:
                retail_price = None

            try:
                year = int(re.findall(r'\d\d\d\d', watch_table.loc['Age', 1])[0])
            except:
                pass

            try:
                wresistance = watch_table.loc['Water resistance', 1]
            except:
                wresistance = 0

            # Retrieving images:
            img = str(watch_soup.find_all('img')[17]).split(',')[1].strip().split(' ')[0]
            img_name = watch_id + '_' + pam + '_' + model
            urllib.request.urlretrieve(img, f'../data/imgs/WF_panerai/{img_name}.jpeg')

            # Wrapping all data to be added to dataframe
            watch = {'id': watch_id,
                     'model': model,
                     'pam': pam,
                     'price': price,
                     'retail_price': retail_price,
                     'year': year,
                     'box': watch_table.loc['Box', 1],
                     'papers': watch_table.loc['Papers', 1],
                     'gender': watch_table.loc['Gender', 1],
                     'movement': watch_table.loc['Movement', 1],
                     'case_size': watch_table.loc['Case size', 1],
                     'case_material': watch_table.loc['Case material', 1],
                     'bracelet_material': watch_table.loc['Bracelet material', 1],
                     'dial_type': watch_table.loc['Dial type', 1],
                     'w_resistance': wresistance,
                     'date': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                     'link': watch_url,
                     'img_path': '../data/imgs/WF_panerai/' + img_name + '.jpeg'
                     }

            print(f'\r{count} / {df_links.size - 1}.- Adding {model}, reference {pam}', end="", flush=True)
            count += 1

            df_features = df_features.append(watch, ignore_index=True)

        except:
            print('\nskipping...')
            pass

        df_features.to_csv('../data/WF_panerai_features.csv')


"""
Updating watches on sale, features and images
"""
# get_links_panerai_on_sale()
# info_and_imgs_panerai_on_sale()
# print(df_features)

