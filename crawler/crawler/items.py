# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PoliticsItem(scrapy.Item):
    content = scrapy.Field()
    def preprocess(self):
        self['content'] = self['content'].split('>')[1].split('>')[0]


