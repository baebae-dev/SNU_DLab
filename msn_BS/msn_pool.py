# # Naver Dictionary Crawling

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
# import argparse
from time import sleep
import json
import re
import sys
import time 

# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('data', type=str, default='train', help='no data')
# parser.add_argument('workers', type=int, default=5, help='no workers')
# args = parser.parse_args()


def crawling_news(urls):
    # print('url', urls, '에 대한 작업 PID', os.getpid())
    url_except = []
    article_ = dict()

    req = requests.get(urls)
    sleep(0.5)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')

    # vert 
    if 'refurl' not in urls:
        url_ = urls.split("/")[4:] 
    else:
        url_ = urls.split("/")[5:]
    article_['verts'] = url_[:2][0]
    # subvert
    article_['subverts'] = url_[:2][1]
    # nid
    nid = url_[-1]
    article_['nids'] = nid.split('?')[0].split('-')[-1]
    
    try:
        # title 
        title = soup.find('header').text.strip()
        article_['titles'] = title
    except:
        article_['titles'] = urls
    try:
        # date
        date = soup.find('time').get('datetime')
        article_['dates'] = date
    except:
        article_['dates'] = urls
    try:
        # content
        content = soup.find('h2').text.strip()
        article_['contents'] = content
    except:
        article_['contents'] = urls
    try:
        # image
        image = soup.find('img').get('src')
        article_['images'] = image
    except:
        article_['images'] = urls

    return article_

if __name__ == '__main__':
    start_time = int(time.time())
    newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_train/news.tsv', delimiter='\t', header=None)
    # newspaper = pd.read_csv('/Users/baeyuna/Documents/SNU_Dlab/Data/MINDlarge_{}/news.tsv'.format(args.data), delimiter='\t', header=None)
    urls = newspaper.iloc[:,5].tolist()

    with Pool(4) as p: 
        news = p.map(crawling_news, urls)

    with open('msn_train_pool.json', 'w') as f:
        json.dump(news, f) 

    print('-------Seconds: %s--------' % (time.time() - start_time))