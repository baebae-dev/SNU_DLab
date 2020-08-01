import scrapy
from msn.items import MsnItem

class MsnSpider(scrapy.Spider):
    name = 'msn'
    allowed_domains = ['https://www.msn.com/en-us/news/world/chile-three-die-in-supermarket-fire-amid-protests/ar-AAJ43pw?ocid=chopendata']
    start_urls = ['https://www.msn.com/en-us/news/world/chile-three-die-in-supermarket-fire-amid-protests/ar-AAJ43pw?ocid=chopendata/']

    def parse(self, response):
        titles = response.xpath('//title/text()')[0].get()
        


        yield    