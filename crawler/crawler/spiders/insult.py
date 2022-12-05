import scrapy
from scrapy_splash import SplashRequest 
from scrapy_selenium import SeleniumRequest
from crawler.items import PoliticsItem, InsultItem
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
import time
display = Display(size=(10, 10))  
display.start()


options = Options()
options.add_argument('--headless')
options.add_experimental_option('excludeSwitches', ['enable-logging'])

driver =  webdriver.Chrome(executable_path="/home/xps/educate/code/hust/DS_20222/data-science-e10/crawler/chromedriver")
driver.get("https://www.reddit.com/r/RoastMe/")

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



"""

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

            for (let i = 0; i < 100; i++) {
                setTimeout(window.scrollBy(0,3000), 2)

            }
"""

class RedditSpider(scrapy.Spider):
    name = 'insult'
    def start_requests(self):

        url  = 'https://www.reddit.com/r/RoastMe/'
        driver.get(url)
        num_scrolls = 200
        for i in range(num_scrolls):
            driver.execute_script("window.scrollBy(0,9000)","")
            time.sleep(0.5)   

        elements = driver.find_elements(By.CSS_SELECTOR, "[data-click-id=body]")
        urls = [element.get_attribute("href") for element in elements]
        for url in urls:
            if len(url) > 10:
                yield SplashRequest(
                            url=url, 
                            callback=self.parse_article,

                        )
        # yield SplashRequest(
        #     url=url,
        #     callback=self.parse,
        #     endpoint='execute',
        #     args={'lua_source': lua_script, 'wait': 5}
        # )
        # yield SeleniumRequest(
        #     url=url,
        #     callback=self.parse,
        #     wait_time=3,
        #     script=sele_script, 
        # )
    def parse_article(self, response):
        # meta = response.meta['splash']['args']['meta']
        # print(meta)   
        item = InsultItem()
        title = response.css("h1").get()
        comments = response.css("[data-testid=comment]").getall()
        item['title'] = title
        item['content'] = '  '.join([comment for comment in comments])


        yield item



    def parse(self, response):
        urls = response.css("[data-click-id=body]::attr(href)").getall()
        for url in urls:
            if len(url) > 10:
                yield SplashRequest(
                    url="https://www.reddit.com" + url, 
                    callback=self.parse_article,

                )
