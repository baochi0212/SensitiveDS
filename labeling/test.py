import json
import os
from glob import glob
processed_path = "/home/xps/educate/code/hust/DS_20222/data-science-e10/data/processed/modelling"
data_labels = ['insult', 'religion', 'terrorism', 'politics']
concat_dict = []
num_samples = {'train': 0.7, 'test': 0.2, 'dev': 0.1} #keep track of splitting
neutral_dict = []
sum_count = 0
for data_label in data_labels:
    f_write = open(f"{processed_path}/{data_label}.json", 'w')
    data_dict = []
    data = json.load(open(f"./{data_label}.json"))
    texts = data.keys()
    labels = data.values()
    count = 0 
    for text, label in zip(texts, labels):
        if len(label) > 0 and 'psychology' not in label:
            count += 1 if 'neutral' in label else 0
            if 'neutral' in label or 'neural' in label:
                neutral_dict.append({'text': text, 'label': 'neutral'})    
            else:
                if data_label in label:
                    data_dict.append({'text': text, 'label': data_label})
                else:
                    data_dict.append({'text': text, 'label': label[0]})
    json.dump(data_dict, f_write, indent=3, ensure_ascii=False)
    num_samples[data_label] = count
    sum_count += count
    f_write.close()
    print("num valid", count)
json.dump(neutral_dict, open(f"{processed_path}/neutral.json", 'w'), indent=3)
#data split
data_labels = []
for filename in glob(f"{processed_path}/*.json"):
    label = os.path.basename(filename)[:-5]
    data_labels.append(label)  
    num_samples[label] = (len(json.load(open(filename, 'r'))))
    concat_dict.extend(json.load(open(filename, 'r')))
sum_count = sum([num_samples[label] for label in num_samples if label not in ['train', 'test', 'dev']])
num_samples = dict([(key, value) if key  in ['train', 'dev', 'test'] else (key, value/sum_count) for key, value in num_samples.items()])


#number of samples:
data_stat = dict([(mode, dict([(label, len(concat_dict)*num_samples[mode]*num_samples[label]) for label in data_labels])) for mode in ['train', 'dev', 'test']])
print(data_stat)
for mode in data_stat.keys():
    data = []
    for file in concat_dict:
        if data_stat[mode][file['label']] > 0:
            data.append(file)
            data_stat[mode][file['label']] -= 1
    json.dump(data, open(f'{processed_path}/sensitive_{mode}.json', 'w'), indent=3)
print("AFTER", data_stat)





