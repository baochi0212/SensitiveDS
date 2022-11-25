import pandas as pd
import re
import json
import os


def json2txt(jsonfile):
    with open(jsonfile, 'r') as f:
        file = json.load(f)
    if not os.path.exists('textfile'):
        os.mkdir('textfile') 
    for i, line in enumerate(file):
        with open(f'textfile/test{i}.txt', 'w') as f:
            f.write(line['content'])

def process(text):
    text = re.findall('\d',  text)
    return text

def parsetxt(file):

    with open(file, 'r') as f:
        with open(file + '_rep.txt') as f_sub:
            for line in f.readlines():
                line = process(line)
                f_sub.write(line)
def test_fn(name: int = 8) -> float:
    '''
    input and return type docs
    '''
    return float(name)
    

if __name__ == '__main__':

    # json2txt('/home/xps/educate/code/hust/DS_20222/data-science-e10/crawler/crawler/spiders/test.json')
    # print(process('<adsfsd bsdbaa ascc'))
    print(test_fn())
