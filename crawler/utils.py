import pandas as pd
import re
import json
import os
from glob import glob
import argparse
raw = "/home/xps/educate/code/hust/DS_20222/data-science-e10/data/raw"
processed = "/home/xps/educate/code/hust/DS_20222/data-science-e10/data/processed"
spiders = "/home/xps/educate/code/hust/DS_20222/data-science-e10/crawler/crawler/spiders"


def json2txt(class_name="politics"):
    with open(spiders + f'/{class_name}.json', 'r') as f:
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
    start = True
    a, b = 1, 1
    while a != -1 and b != -1:
        a, b = string.find('<'), string.find('>')
    
        c = b + string[b+1:].find('<')
        if a != 0 and start:
            new_string = string[:a]
            start = False
        if c > 0:
            join_char = ' '
            new_string = join_char.join([new_string.strip(), string[b+1:c].strip()])
        else:
            join_char = ' '
            new_string = join_char.join([new_string.strip(), string.strip()])
        string = string[b+1:]
    return new_string.strip()
def parse_folder(class_name="politics", min_length=5):
    def miniprocess(sentence):
        sentence = sentence.replace(r'”', '')
        sentence = sentence.replace(r'“', '')
        sentence = sentence.replace(r'"', '')
        sentence = sentence.strip()
        return sentence
    path = raw + f"/{class_name}/*.txt"
    files = glob(path)
    num_text = 0
    for file in files:
        path = f"{file}"
        text = open(path, 'r').readline()
        text = parse(text)
        with open(processed + f"/{class_name}.txt", 'a') as f: 
            # text = '. '.join(text.split('..'))
            text = text.replace('U.S', 'US')
            text = text.replace('.,', ',')
            num_sentence = 0
            for i in range(len(text.split('.'))):
                sentence = text.split('.')[i]
                count = 1
                if len(sentence.split()) < min_length:
                    continue
                while i + count < len(text.split('.')) and len(text.split('.')[i+count].split()) < min_length:
                    sentence = '. '.join([sentence, text.split('.')[i+count]])
                    count += 1
                sentence = miniprocess(sentence)
              

                f.write(sentence + '\n')
                num_sentence += 1
                num_text += 1 
            print(f"FILE {file} with num {num_sentence}")



    print("NUMBER of TEXT", num_text)



if __name__ == '__main__':

    # json2txt('/home/xps/educate/code/hust/DS_20222/data-science-e10/crawler/crawler/spiders/crawl.json')
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", required=True)
    parser.add_argument("--spider_type", required=True)
    args = parser.parse_args()
    json2txt(class_name=args.class_name)
    parse_folder(class_name=args.class_name)

