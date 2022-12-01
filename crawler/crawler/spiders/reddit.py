import scrapy
from scrapy_splash import SplashRequest 
from scrapy_selenium import SeleniumRequest
from crawler.items import PoliticsItem, InsultItem
import os
import pandas as pd

# # data_path = os.environ['data_path']

lua_script = """
        function main(splash)
            local num_scrolls = 100
            local scroll_delay = 30
            local num_clicks = 4
            local click_delay = 0.5
            local scroll_to = splash:jsfunc("window.scrollTo")
            local get_body_height = splash:jsfunc(
                "function() {return document.body.scrollHeight;}"
            )
            local url = splash.args.url
            assert(splash:go(url))
            splash:wait(splash.args.wait)

            for _ = 1, num_scrolls do
                scroll_to(0, get_body_height())
                splash:wait(scroll_delay)
             

                
                
            end        
            return {
                html = splash:html(),
                url = splash:url()
                }
                
        end



#         """

# title_script = """
#         function main(splash)
#             local num_scrolls = 20
#             local scroll_delay = 0.5
#             local num_clicks = 4
#             local click_delay = 1
#             local scroll_to = splash:jsfunc("window.scrollTo")
#             local get_body_height = splash:jsfunc(
#                 "function() {return document.body.scrollHeight;}"
#             )
#             local url = splash.args.url
#             assert(splash:go(url))
#             splash:wait(splash.args.wait)




       
             

                
                
#             end        
#             return {
#                 html = splash:html(),
#                 url = splash:url()
#                 }
                
#         end



#         """

sele_script = """

            for (let i = 0; i < 10; i++) {
                window.scrollBy(0,1000)

            }
"""

class RedditSpider(scrapy.Spider):
    name = 'test'
    def start_requests(self):

        url  = 'https://www.reddit.com/r/RoastMe/'
        # yield SplashRequest(
        #     url=url,
        #     callback=self.parse,
        #     args={'lua_source': lua_script, 'wait': 100}
        # )
        yield SeleniumRequest(
            url=url,
            callback=self.parse,
            wait_time=3,
            script=sele_script, 
        )
    def parse_article(self, response):
        # meta = response.meta['splash']['args']['meta']
        # print(meta)   
        item = InsultItem()
        comments = response.css("[data-testid=comment]").getall()
        item['content'] = '@'.join([comment for comment in comments])


        yield item



    def parse(self, response):
        urls = response.css("[data-click-id=body]::attr(href)").getall()
        # yield item
        for url in urls:
            if len(url) > 10:
                yield SplashRequest(
                    url="https://www.reddit.com" + url, 
                    callback=self.parse_article,

                )
