import scrapy
from msn_large.items import MsnLargeItem

class MsnLargeSpider(scrapy.Spider):
    name = 'msn_large'
    allowed_domains = ['assets.msn.com/labs/mind/AAGH0ET.html']
    start_urls = ['http://assets.msn.com/labs/mind/AAGH0ET.html/']

    def parse(self, response):
        titles = response.css('body>div>header>h1::text').get().strip()
        dates = response.css('section.Modelinfo>div>div>div>span.date>time::text').get().strip()
        images = response.css('#main > article > section > ul > li:nth-child(1) > div > img::attr("src")').get()
        abstracts = response.xpath('//*[@id="main"]/article/div[2]/div[1]/div[2]/div/h2/text()').get().strip()
        contents = ' '.join(response.css('#main > article > div.gallerydata > div.show > div.body-text > div > figure > figcaption > div > div > p::text').getall())
        
        print('titles :', titles)
        print('dates :', dates)
        print('images :', images)
        print('abstracts :', abstracts) 
        print('contents :', contents) 

        item = MsnLargeItem()
        item['title'] = titles # item에 넣어라 
        item['dates'] = dates
        item['images'] = images
        item['abstracts'] = abstracts
        item['contents'] = contents

        yield item 


