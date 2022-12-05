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
# #scroll by pixel
# num_scrolls = 4
# for i in range(num_scrolls):
#     driver.execute_script("window.scrollBy(0,3000)","")
#     time.sleep(10)
# elements = driver.find_elements(By.CSS_SELECTOR, "[data-click-id=body]")
# elem = [element.get_attribute("href") for element in elements]
# print("elements", len(elem))

#url
driver.get("https://www.reddit.com/r/RoastMe/")

num_scrolls = 20
for i in range(num_scrolls):
    driver.execute_script("window.scrollBy(0,3000)","")
    time.sleep(0.5)   

elements = driver.find_elements(By.CSS_SELECTOR, "[data-click-id=body]")
elem = [element.get_attribute("href") for element in elements]
print("elements", len(elem))