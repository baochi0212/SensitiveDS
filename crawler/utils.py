import pandas as pd

df = pd.read_csv('/home/xps/educate/code/hust/DS_20222/crawler/crawler/spiders/test.csv')
import json
import os
def json2txt(jsonfile):
    with open(jsonfile, 'r') as f:
        file = json.load(f)
        print(file[0]['content'])
    if not os.path.exists('textfile'):
        os.mkdir('textfile') 
    for i, line in enumerate(file):
        with open(f'textfile/test{i}.txt', 'w') as f:
            f.write(line['content'])


json2txt('/home/xps/educate/code/hust/DS_20222/crawler/crawler/spiders/test.json')