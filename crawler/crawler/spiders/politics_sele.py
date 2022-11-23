import scrapy
from scrapy_splash import SplashRequest 
from scrapy_selenium import SeleniumRequest
from crawler.items import PoliticsItem, TestItem
import os
import pandas as pd

data_path = os.environ['data_path']

content_script = """
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



        """

title_script = """
        function main(splash)
            local num_scrolls = 20
            local scroll_delay = 0.5
            local num_clicks = 4
            local click_delay = 1
            local scroll_to = splash:jsfunc("window.scrollTo")
            local get_body_height = splash:jsfunc(
                "function() {return document.body.scrollHeight;}"
            )
            local url = splash.args.url
            assert(splash:go(url))
            splash:wait(splash.args.wait)




       
             

                
                
            end        
            return {
                html = splash:html(),
                url = splash:url()
                }
                
        end



        """

sele_script = """

            for (let i = 0; i < 5; i++) {
                document.querySelector('div.styles_loadMoreWrapper__pOldr button').click()

            }
"""

class PoliticsSpider(scrapy.Spider):
    name = 'test'
    def start_requests(self):

        url  = 'https://www.nbcnews.com/politics'
        yield SeleniumRequest(
            url=url,
            callback=self.parse,
            script=sele_script,
            wait_time=3,
        )
    def parse_article(self, response):
        # meta = response.meta['splash']['args']['meta']
        # print(meta)   
        item = PoliticsItem()
        title = response.css('h1::text').get()
        contents = ' '.join([i for i in response.css('div.article-body__content').css('p').getall()])
        item['title'] = title
        item['content'] = contents


        # item.preprocess()
        if title and contents:
            yield item
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

    #         # keep passing the metadata and used parser to next request
    #     df = pd.DataFrame.from_dict(dict([(k, [None]) for k in temp_item.keys()]))

        
    #     while len(df) != len(urls):  
            
    #         self.logger.info("NOT FINISHED YET")          
    #         for item in urls:
    #             if item['title_raw'] not in df['title_raw']:

    #                 yield SplashRequest(
    #                     item['url'], 
    #                     callback=self.parse_article, 
    #                     endpoint='execute', 
    #                     args={'wait': 0.5, 'lua_source': lua_script, 'meta': item} 
    #                     )
    #         df = pd.read_csv(data_path)
        # #trace
        # while len(pd.read_csv(test_path)) != len(urls):
        #     print("?????")
        #     self.logger.info("NOT FINISHED YET")
        #     df = pd.read_csv(test_path)
        #     for item in urls:
        #         if item['title_raw'] not in df['title_raw']:
        #             yield SplashRequest(
        #                 item['url'], 
        #                 callback=self.parse_article, 
        #                 endpoint='execute', 
        #                 args={'wait': 0.5, 'lua_source': lua_script, 'meta': item})
        # print("FINISHED ?")