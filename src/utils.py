import os
from glob import glob
import json



dir_path = os.environ['dir']
labeling_path = f"{dir_path}/labeling"
processed_path = f"{dir_path}/data/processed"
def json_format():
    def process(text):
        return text
    #isolate neutral
    neutral_dict = []
    for file in glob(f"{labeling_path}/*.json"):
        basename = os.path.basename(file)
        data = json.load(open(file, 'r'))
        texts = data.keys()
        labels = data.values()
        data_dict = []
        count = 0 
        for text, label in zip(texts, labels):

            text = process(text)
            if len(label) > 0 and label[0] != "neutral":
                data_dict.append({'text': text, 'label': label})
                count += 1 
            elif len(label) > 0: 
                neutral_dict.append({'text': text, 'label': label})
                
        
        json.dump(data_dict, open(f"{processed_path}/{basename}", 'w'), indent=3, ensure_ascii=False)
        print(f"num valid {basename}", count)
    json.dump(neutral_dict, open(f"{processed_path}/neutral.json", 'w'), indent=3, ensure_ascii=False)
    print(f"num valid neutral", len(neutral_dict))

if __name__ == '__main__':
    json_format()
        