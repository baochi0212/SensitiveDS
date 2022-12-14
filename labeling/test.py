import json
import os

labels = ['insult', 'religion', 'terrorism']
for label in labels:
    # data = json.load(open(f"./{label}.json"))
    # texts = data.keys()
    # labels = data.values()
    # count = 0 
    # for text, label in zip(texts, labels):
    #     # print(label)
    #     if len(label) > 0:
    #         count += 1 if label[0] != 'neutral' else 0

    # print("num valid", count)
    print(os.path.basename(f"./{label}.json"))
