import scrapy
from msn.items import MsnItem

class MsnSpider(scrapy.Spider):
    name = 'msn'
    allowed_domains = ['https://www.msn.com/en-us/news/world/chile-three-die-in-supermarket-fire-amid-protests/ar-AAJ43pw?ocid=chopendata']
    start_urls = ['https://www.msn.com/en-us/news/world/chile-three-die-in-supermarket-fire-amid-protests/ar-AAJ43pw?ocid=chopendata/']

    def parse(self, response):
        titles = response.xpath('//title/text()')[0].get()
        # dates = response.css('selection > div.authorinfo-flexar > div > div > span[3] > time::text').get()
        # images = response.css('article > section.flexarticle > section.articlebody >  span.image > img::attr("src")').get() 

        for num, title in enumerate(titles):
            item = MsnItem() # 클래스로 객체 생성 
            item['title'] = title # item에 넣어라 
            yield item # yield 할때마다 items.py의 item 객체에 데이터가 쌓인다. 