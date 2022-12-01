import scrapy
from scrapy_splash import SplashRequest 
from scrapy_selenium import SeleniumRequest
from crawler.items import PoliticsItem, TestItem
import os
import pandas as pd

# # data_path = os.environ['data_path']

lua_script = """
        function main(splash)
            local num_scrolls = 20
            local scroll_delay = 0.5
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

            for (let i = 0; i < 3; i++) {
                document.querySelector('div.styles_loadMoreWrapper__pOldr button').click()

            }
"""

class PoliticsSpider(scrapy.Spider):
    name = 'test'
    def start_requests(self):

        # url  = 'https://xe.chotot.com/mua-ban-xe-may-quan-lien-chieu-da-nang/99205208.htm#px=SR-stickyad-[PO-1][PL-top]'
        url = r"https://xe.chotot.com/"
        # yield SeleniumRequest(
        #     url=url,
        #     callback=self.parse,
        #     script=sele_script,
        #     wait_time=3,
        # )
        yield SplashRequest(
            callback=self.parse, 
            endpoint='execute', 
            args={'wait': 10, 'lua_source': lua_script, 'url': url}
            )


    def parse(self, response):
        urls = response.css('div.styles_itemsContainer__saJYW').css('a::attr(href)').getall()
        button = response.css('div.styles_loadMoreWrapper__pOldr button').get()
        # yield item
        for url in urls:
            if len(url) > 100:
                yield SplashRequest(
                    url=url, 
                    callback=self.parse_article,
                )
