from nltk.corpus import inaugural, stopwords, wordnet
from nltk.wsd import lesk
from nltk import word_tokenize, sent_tokenize
import numpy as np
import json


stop_words = set(stopwords.words('english'))


def syn_to_lem(key_synsets):
    words = set()
    for syns in key_synsets:
        syns = wordnet.synset(syns)
        words.update(list(map(lambda x:' '.join(x.name().split('_')).lower(), syns.lemmas())))
    return words

with open('..\\data\\keyword_group.json') as f:
    keywords = json.load(f)
keywords = dict((x.split('.')[0], syn_to_lem(keywords[x])) for x in keywords.keys())
key_synsets = list(keywords.keys())

def clean_corpus(corpus):
    return list(map(lambda sent:list(filter(lambda word:not word in stop_words, sent)), corpus))

def find_features(corpus_dict): #corpus: dict of year:[list of sents]
    features_dict = dict()
    for fileid in corpus_dict:
        corpus = clean_corpus(corpus_dict[fileid])
        total_len = sum(map(len, corpus))
        ind = fileid.split('.')[0]
        Synset_usage = dict((x, 0) for x in key_synsets)
        for sent in corpus:
            sent = ' '.join(sent).lower()
            for key, pool in keywords.items():
                count = 0
                for w in pool:
                    if w in sent:
                        count+=1
                Synset_usage[key] +=count/total_len
        features_dict[ind] = Synset_usage
    return features_dict

def process_bodytext(text):
    text = sent_tokenize(text)
    return list(map(word_tokenize, text))

def main():

    inaug = dict((fileid, inaugural.sents(fileid)) for fileid in filter(lambda x:int(x[:4])>=1960, inaugural.fileids()))
    features_dict = find_features(inaug) #The word the President said and its amount
    print(features_dict)
    with open('..\\data\\inaugural_scores.json', 'w') as f:
        json.dump(features_dict, f)

    with open('..\\data\\sotus_a.json') as f:
        sotus = json.load(f)
    sotus = dict((x['date'][-4:], process_bodytext(x['body'])) for x in sotus)
    features_dict = find_features(sotus)
    print(features_dict)
    with open('..\\data\\sotus_scores.json', 'w') as f:
        json.dump(features_dict, f)

if __name__ == '__main__':
    main()
