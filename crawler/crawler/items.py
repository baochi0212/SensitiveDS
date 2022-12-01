# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

class TestItem(scrapy.Item):
    url = scrapy.Field()
    button = scrapy.Field()

class PoliticsItem(scrapy.Item):
    title = scrapy.Field()
    content = scrapy.Field()
    
    def preprocess(self):
        self['content'] = self['content'].split('>')[1].split('>')[0]

class InsultItem(scrapy.Item):
    title = scrapy.Field()
    content = scrapy.Field()


