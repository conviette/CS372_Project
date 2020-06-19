from nltk.corpus import inaugural, stopwords, wordnet
from nltk import word_tokenize, sent_tokenize, FreqDist, DictionaryProbDist
import numpy as np
import json, re
from project_dataset import textData

##calculating features for each year

##get stopwords
stop_words = set(stopwords.words('english'))

STAT_NAME = 'GDP_GROWTH' ##'GDP_GROWTH' or 'EXPORT_INDEX' or 'UNEMPLOYMENT'


def read_keywords():
    with open('..\\Data\\{}_found_features.json'.format(STAT_NAME)) as f:
        d = json.load(f)
    unigrams = list(filter(lambda x:x.count('_')==0, d))
    bigrams = list(filter(lambda x:x.count('_')==1, d))
    return unigrams, bigrams

def clean_corpus(corpus): #any preprocessing for corpus
    corpus = list(map(lambda sent:list(filter(lambda word:(not word in stop_words) and len(word)>3, sent)), corpus))
    return [[word.lower() for word in sent] for sent in corpus]#discarding stopwords

#####functions for computing each features#########
#####all functions must accept feature_dict as input, and add entry "feature_name" mapped to number

key_uni, key_bi = read_keywords()

def find_features_agg(corpus_dict):
    features_for_year = dict()

    for fileid in corpus_dict:
        features_for_year[fileid] = dict()
        corpus = clean_corpus(corpus_dict[fileid])
        vocabulary_unigram = FreqDist([word for sent in corpus for word in sent])
        unigram_pdist = DictionaryProbDist(vocabulary_unigram, normalize=True)
        for key in key_uni:
            if key in vocabulary_unigram:
                features_for_year[fileid][key] = unigram_pdist.prob(key)
            else:
                features_for_year[fileid][key] = 0.0

        big_corpus = [s[i]+'_'+s[i+1] for s in corpus for i in range(len(s)-1)]
        vocab_bigram = FreqDist(big_corpus)
        bigram_pdist = DictionaryProbDist(vocab_bigram, normalize=True)
        for key in key_bi:
            if key in vocab_bigram:
                features_for_year[fileid][key] = bigram_pdist.prob(key)
            else:
                features_for_year[fileid][key] = 0.0

    return features_for_year #output features_for_year: dict {year:{feature_name:float}}



def main():

    ###Inaugural corpus
    inaug = textData('INAUGURAL').corpus
    features_dict = find_features_agg(inaug) #The word the President said and its amount
    #print(features_dict)
    with open('..\\data\\inaugural_{}_test_scores.json'.format(STAT_NAME), 'w') as f:
        json.dump(features_dict, f)

    ###State of the Union Speech
    sotus = textData('SOTUS').corpus
    features_dict = find_features_agg(sotus)
    #print(features_dict)
    with open('..\\data\\sotus_{}_test_scores.json'.format(STAT_NAME), 'w') as f:
        json.dump(features_dict, f)

    ###All Oral Speeches
    oral = textData('ORAL').corpus
    features_dict = find_features_agg(oral)
    #print(features_dict)
    with open('..\\data\\oral_{}_test_scores.json'.format(STAT_NAME), 'w') as f:
        json.dump(features_dict, f)


if __name__ == '__main__':
    main()
