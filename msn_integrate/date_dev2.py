import bs4
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import pandas as pd 
from tqdm import tqdm
from random import *
import time 

options = Options()
urls = []

# not_tr : 5733
# not_dev : 8186

with open("not_dev.csv", "r") as file:
    i = 0
    for line in file:
        if 1000 < i < 2000:
            stripped_line = line.replace('"', '').strip()
            urls.append(stripped_line)
        i += 1


# Don't open browser 
options.add_argument("--headless")

unparseable_urls = []
dates_lst = []
urls_lst = []
print('crawling start.')
start_time = int(time.time())
MAX_SLEEP_TIME = 3 
# start web browser 
chrome_browser = webdriver.Chrome(executable_path='/Users/baeyuna/Documents/chromedriver', options=options)
with chrome_browser as browser:
    print("bottleneck .... Program will open chrome ... takes ages ...")
    for url in tqdm(urls):
        rand_value = randint(1, MAX_SLEEP_TIME)
        time.sleep(rand_value)
        chrome_browser.get(url)
        try:
            date_element = chrome_browser.find_elements_by_tag_name('time')[0]
            date = date_element.get_attribute('datetime')
            dates_lst.append(date)
            # print(date)

            urls_lst.append(url)
        except:
            unparseable_urls.append(url)
            # print(f"Yeee!!! Cannot parse from: {url}")
    # close web browser to close session and avoid socket errors
    browser.quit()

# # Unparseable url log
# with open("unparseable.txt", 'a+') as log:
#     output = ""
#     for failed_url in unparseable_urls:
#         output += failed_url 
#         output += "\n"
#     log.write(output)

print('-----unparseable_urls-------')
print(len(unparseable_urls))

print('-------Seconds: %s--------' % (time.time() - start_time))
df = pd.DataFrame({'date':dates_lst, 'url': urls_lst})
print(df.shape)
df.to_csv('date_dev2.csv', index=False) 
print('date_dev2.csv saved')