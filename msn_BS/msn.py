# # Naver Dictionary Crawling

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
import argparse
import json
import re
import sys

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('data', type=str, help='no data')
parser.add_argument('save_format', type=str, help='')
args = parser.parse_args()

verts = []
subverts = []
nids = []
titles = []    
dates = []
contents = []
images = []

newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_{}/news.tsv'.format(args.data), delimiter='\t', header=None)
urls = newspaper.iloc[:,5]

for url in urls:
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

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
    nids.append(nid.split('?')[0].split('-')[-1])
    # title
    title = soup.find('header').text.strip()
    titles.append(title)
    # date
    date = soup.find('time').get('datetime')
    dates.append(date)
    # content
    content = soup.find('h2').text.strip()
    contents.append(content)
    # image
    image = soup.find('img').get('src')
    images.append(image)


df = pd.DataFrame({'verts':verts,
                   'subverts': subverts,
                   'nids': nids,
                   'Title':titles,
                   'Date': dates, 
                   'Content': contents,
                   'Image': images})

# 데이터 저장
if args.save_format == 'csv' :
    df = df.to_csv('msn_{}.csv'.format(args.data), index=False)
elif args.save_format == 'json' :
    with open('msn_{}.json'.format(args.data), 'w') as f:
        json.dump(df, f)