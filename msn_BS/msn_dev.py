# # Naver Dictionary Crawling

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import json
import re
import sys
import time


url_lst = []
nids = []
titles = []    
dates = []
contents = []
images = []

newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_dev/news.tsv', delimiter='\t', header=None)
urls = newspaper.iloc[:,5]

print('crawling start.')
start_time = int(time.time())

url_except =[]
num = 0
total = len(urls)
for url in urls:
    # 진행상황 확인 
    num += 1
    if num % 5000 == 0:
        print('{} urls are crawled.\n {} are remained'.format(num, total - num))

    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    # url
    url_lst.append(url)

    # title
    if soup.find('h1').text.strip() is not None:
        title = soup.find('h1').text.strip()
        titles.append(title)
    elif soup.find('header').text.strip() is not None:
        title = soup.find('header').text.strip()
        titles.append(title)
    else:
        titles.append(None)


    # date
    try:
        date = soup.find('time').get('datetime')
        dates.append(date)
    except:
        dates.append(url)

    # content
    if soup.find_all('p') is not None:
        content = []
        for el in soup.find_all('p'):
            content.append(el.get_text()) 
        contents.append(' '.join(content))
    elif soup.find('h2').text.strip() is not None:
        content = soup.find('h2').text.strip()
        contents.append(content)
    else:
        contents.append(url)


    # image
    try:
        image = soup.find('img').get('src')
        images.append(image)
    except:
        images.append('no image')

    # nid
    nid = url.split('/')[-1].split('.')[0]
    nids.append(nid)


df = pd.DataFrame({'url' : url_lst,
                   'nids': nids,
                   'Title':titles,
                   'Date': dates, 
                   'Content': contents,
                   'Image': images})

# 데이터 저장   
print('-------df--------')
print(df.shape)
df.to_csv('msn_dev.csv', index=False)

print('-------Seconds: %s--------' % (time.time() - start_time))