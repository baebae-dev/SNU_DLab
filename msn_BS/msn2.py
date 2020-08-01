# # Naver Dictionary Crawling

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import json
import re
import sys

url_lst = []
verts = []
subverts = []
nids = []
titles = []    
dates = []
contents = []
images = []

newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_train/news.tsv', delimiter='\t', header=None)
urls = newspaper.iloc[:,5]

print('crawling start.')
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
    try:
        title = soup.find('header').text.strip()
        titles.append(title)
    except:
        title = soup.find('h1').text.strip()
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
    try:
        content = soup.find('h2').text.strip()
        contents.append(content)
    except:
        content = []
        for el in soup.find_all('p'):
            content.append(el.get_text()) 
        contents.append(' '.join(content))
    else:
        contents.append(url)

    # image
    try:
        image = soup.find('img').get('src')
        images.append(image)
    except:
        images.append('no image')

    # vert
    if 'refurl' not in url:
        url = url.split("/")[4:] 
    else:
        url = url.split("/")[5:]
    verts.append(url[:2][0])

    # subvert
    subverts.append(url[:2][1])
    
    # nid
    nid = url[-1]
    nids.append(nid.split('?')[0].split('-')[-1].split('.')[0])


print('-----url_except-----')
print(len(url_except))
df = pd.DataFrame({'verts':verts,
                   'subverts': subverts,
                   'nids': nids,
                   'Title':titles,
                   'Date': dates, 
                   'Content': contents,
                   'Image': images})

# 데이터 저장   
print('-------df--------')
print(df.shape)
df.to_csv('msn_train.csv', index=False)

