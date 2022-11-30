import pandas as pd
import re
import json
import os
from glob import glob
raw = "/home/xps/educate/code/hust/DS_20222/data-science-e10/data/raw"
processed = "/home/xps/educate/code/hust/DS_20222/data-science-e10/data/processed"


def json2txt(jsonfile, class_name="politics"):
    with open(jsonfile, 'r') as f:
        file = json.load(f)
    for i, line in enumerate(file):
        with open(f'{raw}/{class_name}/{i}.txt', 'w') as f:
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

def parse(string):
    new_string = ""
    a, b = 1, 1
    while a != -1 and b != -1:
        a, b = string.find('<'), string.find('>')
    
        c = b + string[b+1:].find('<')
        if c > 0:
            new_string = ' '.join([new_string, string[b+1:c]])
        else:
            new_string = ' '.join([new_string, string[b+1:]])
        string = string[b+1:]
    return new_string.strip()
def parse_folder(class_name="politics"):

    path = raw + f"/{class_name}/*.txt"
    files = glob(path)
    num_text = 0
    for file in files:
        path = f"{file}"
        text = open(path, 'r').readline()
        text = parse(text)
        
        for sentence in text.split('.'):
            with open(processed + f"/{class_name}/{num_text}.txt", 'w') as f: 
                f.write(sentence.strip())
                num_text += 1 
    print("NUMBER of TEXT", num_text)



if __name__ == '__main__':

    json2txt('/home/xps/educate/code/hust/DS_20222/data-science-e10/crawler/crawler/spiders/crawl.json')
    parse_folder(class_name="politics")
    # print(process('<adsfsd bsdbaa ascc'))

