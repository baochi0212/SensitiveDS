import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import json
from copy import deepcopy
import random


class Augmenter:
    def __init__(self, options, n=1):
        self.options = options
        self.n = n

    def back_translate(self, text):
        back_translation_aug = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de', 
            to_model_name='facebook/wmt19-de-en'
        )
        return back_translation_aug.augment(text, n=self.n)
    def embedding(self, text):
        augmentor = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')
        return augmentor.augment(text, n=self.n) 

    def __call__(self, data_dict):
        new_dict = deepcopy(data_dict)
        for data in data_dict:
            text, label = data['text'], data['label']
            if 'backtranslate' in self.options:
                if random.uniform(0, 1) > 0.7:
                    back_translation = [dict([("text", t), ("label", label)])
                                    for t in self.back_translate(text)]     
                    new_dict.extend(back_translation)
            if 'embedding' in self.options:
                embedding = [dict([("text", t), ("label", label)])
                                for t in self.embedding(text)]

            
                new_dict.extend(embedding)
        return new_dict     

if __name__ ==  '__main__':
    data_path = "/home/tranbaochi_/Study/hust/data-science-e10/source/data/sensitive_train.json"
    augment_path = "/home/tranbaochi_/Study/hust/data-science-e10/source/data/sensitive_augment.json"
    init_augment = Augmenter(options=['backtranslate']) 
    data_dict = json.load(open(data_path, 'r'))
    #augment
    augment_dict = init_augment(data_dict)     
    print(len(data_dict), len(augment_dict))