# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class MsnLargeItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    dates = scrapy.Field()
    titles = scrapy.Field()
    contents = scrapy.Field()
    images = scrapy.Field()
