import scrapy
from urllib.parse import urlencode

API_KEY = '44898f0a-c6a5-4818-a7b0-130947187143'

def get_scrapeops_url(url):
    payload = {'api_key': API_KEY, 'url': url, 'bypass': 'cloudflare'}
    proxy_url = 'https://proxy.scrapeops.io/v1/?' + urlencode(payload)
    return proxy_url

class testSpider(scrapy.Spider):
    name = "test_scrapy_quotes"

    def start_requests(self):
        urls = [
            'https://www.mirror.co.uk/all-about/suicide'
        ]
        for url in urls:
            yield scrapy.Request(callback=self.parse)
