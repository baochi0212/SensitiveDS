import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import json
from copy import deepcopy
import random
import os

main_dir = os.environ.get('MAIN_DIR')

class Augmenter:
    def __init__(self, options, n=1):
        self.options = options
        self.n = n
        self.back_translation_aug = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en'
        )
        self.embedding = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')

    def back_translate(self, text):
  
        return self.back_translation_aug.augment(text, n=self.n)
    def embed(self, text):
        
        return self.embedding.augment(text, n=self.n) 

    def __call__(self, data_dict):
        new_dict = deepcopy(data_dict)
        for data in data_dict:
            text, label = data['text'], data['label']
            if 'embedding' in self.options:
                if random.uniform(0, 1) > 0.3:
                    embedding = [dict([("text", t), ("label", label)])
                                for t in self.embed(text)]
                    new_dict.extend(embedding)
            if 'backtranslate' in self.options:
                
                if random.uniform(0, 1) > 0.7:
                    back_translation = [dict([("text", t), ("label", label)])
                                    for t in self.back_translate(text)]     
                    new_dict.extend(back_translation)
            

            
                
        return new_dict     

if __name__ ==  '__main__':
    data_path = main_dir + '/source/data/sensitive_train.json' 
    augment_path = main_dir + "/source/data/sensitive_augment.json"
    init_augment = Augmenter(options=['backtranslate', 'embedding'], n=1) 
    data_dict = json.load(open(data_path, 'r'))
    #augment
    augment_dict = init_augment(data_dict)   
    f_write = open(augment_path, 'w')
    json.dump(augment_dict, f_write, indent=3)  
    print(len(data_dict), len(augment_dict))