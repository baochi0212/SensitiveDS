import scrapy
from scrapy_splash import SplashRequest 
from crawler.items import PoliticsItem
import os
import pandas as pd

data_path = os.environ['data_path']

lua_script = """
        function main(splash)
            local num_scrolls = 20
            local scroll_delay = 1.0

            local scroll_to = splash:jsfunc("window.scrollTo")
            local get_body_height = splash:jsfunc(
                "function() {return document.body.scrollHeight;}"
            )
            local url = splash.args.url
            assert(splash:go(url))
            splash:wait(splash.args.wait)

            for _ = 1, num_scrolls do
                scroll_to(0, get_body_height())urls = response.css('div.styles_itemsContainer__saJYW').css('a::attr(href)').getall()
                splash:wait(scroll_delay)
                btn = splash:select_all('.animated-ghost-button animated-ghost-button--normal styles_button__khb8K')[0]
                btn:mouse_click()
                splash:wait(scroll_delay)
                
                
            end        
            return {
                html = splash:html(),
                url = splash:url()
                }
                
        end



        """

class TestSpider(scrapy.Spider):
    name = 'test'
    def start_requests(self):

        url  = 'https://www.nbcnews.com/politics'
        yield SplashRequest(
            url, 
            callback=self.parse, 
            endpoint='execute', 
            args={'wait': 2.5, 'lua_source': lua_script}
            )
    def parse_article(self, response):
        # meta = response.meta['splash']['args']['meta']
        item = PoliticsItem()
        # title = response.css('h1::text').get()
        contents = ' '.join([i for i in response.css('div.article-body__content').css('p').getall()])
        item['content'] = contents


        # item.preprocess()
        if item and contents:
            yield item
    def parse(self, response):
        urls = response.css('div.styles_itemsContainer__saJYW').css('a::attr(href)').getall()
        for url in urls:
            if len(url) > 100:
                yield SplashRequest(
                    url, 
                    callback=self.parse_article,
                    endpoint='execute',
                    args={'wait': 0.5, 'lua_source': lua_script} 
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