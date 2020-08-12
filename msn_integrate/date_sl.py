from tqdm import tqdm 
import bs4
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import time
from random import *
import pandas as pd 
import argparse

parser = argparse.ArgumentParser(description='date crawling url index')
parser.add_argument('server', type=int)
args = parser.parse_args()

def date_crawler(urls):
    MAX_SLEEP_TIME = 3 

    options = Options()
    # Don't open browser
    options.add_argument("--headless")
    # start web browser
    chrome_browser = webdriver.Chrome(executable_path='/Users/baeyuna/Documents/chromedriver', options=options)

    dates = []
    with chrome_browser as browser:
        print("bottleneck .... Program will open chrome ... takes ages ...")
        # get source code
        for url in tqdm(urls): 
            rand_value = randint(1, MAX_SLEEP_TIME)
            time.sleep(rand_value)
            browser.get(url)
            print("that was slow ....... ") 
            html = browser.page_source 
            soup = bs4.BeautifulSoup(html, 'html.parser')
            # Find datetime
            for i in soup.findAll('time'): 
                if i.has_attr('datetime'):
                    dates.append(i['datetime'])
            # close web browser to close session and avoid socket errors
            browser.quit() 

    df = pd.DataFrame({'date':dates, 'url':urls})
    return df

if __name__ == "__main__":
    # tr 추가 수집 링크
    not_tr = pd.read_csv('not_tr.csv')
    not_tr_urls = not_tr['urls'].tolist() 
    # dev 추가 수집 링크
    not_dev = pd.read_csv('not_dev.csv')
    not_dev_urls = not_dev['urls'].tolist()

    # 서버별 수집 링크 할당
    if args.server == 'warhol1':
        urls = not_tr_urls[:2000]
    elif args.server == 'warhol2':
        urls = not_tr_urls[2000:4000]
    elif args.server == 'warhol3':
        urls = not_tr_urls[4000:]
    elif args.server == 'warhol4':
        urls = not_dev_urls[:2000]
    elif args.server == 'warhol5':
        urls = not_dev_urls[2000:4000]
    elif args.server == 'warhol6':
        urls = not_dev_urls[4000:6000]
    elif args.server == 'warhol7':
        urls = not_dev_urls[6000:]
    
    df = date_crawler(urls)
    if args.server == 'warhol1' or 'warhol2' or 'warhol3':
        df.to_csv('integrated_news_tr_{}'.format(args.server))
    else:
        df.to_csv('integrated_news_dev_{}'.format(args.server))