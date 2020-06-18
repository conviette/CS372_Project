from nltk.corpus import inaugural, stopwords, wordnet
from nltk import word_tokenize, sent_tokenize, FreqDist, DictionaryProbDist
import numpy as np
import json, re
from project_dataset import textData

##calculating features for each year

##get stopwords
stop_words = set(stopwords.words('english'))

def clean_corpus(corpus): #any preprocessing for corpus
    corpus = list(map(lambda sent:list(filter(lambda word:(not word in stop_words) and len(word)>3, sent)), corpus))
    return [[word.lower() for word in sent] for sent in corpus]#discarding stopwords

#####functions for computing each features#########
#####all functions must accept feature_dict as input, and add entry "feature_name" mapped to number

def find_features(corpus_dict): #input corpus_dict: dict {year:[list of sents]}
    features_for_year = dict()
    keywords = set()
    for fileid in corpus_dict: #for each year
        corpus = clean_corpus(corpus_dict[fileid]) #clean corpus
        vocabulary_unigram = FreqDist([word for sent in corpus for word in sent])
        unigram_pdist = DictionaryProbDist(vocabulary_unigram, normalize=True)
        unigram_pdist = dict((x[0], unigram_pdist.prob(x[0])) for x in vocabulary_unigram.most_common(100))
        for key in unigram_pdist:
            keywords.add(key)
        features_for_year[fileid] = unigram_pdist
    for d in features_for_year.values():
        for key in keywords:
            if not key in d:
                d[key] = 0.0
    return features_for_year #output features_for_year: dict {year:{feature_name:float}}



def main():

    ###Inaugural corpus
    inaug = textData('INAUGURAL').corpus
    features_dict = find_features(inaug) #The word the President said and its amount
    #print(features_dict)
    with open('..\\data\\inaugural_train_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###State of the Union Speech
    sotus = textData('SOTUS').corpus
    features_dict = find_features(sotus)
    #print(features_dict)
    with open('..\\data\\sotus_train_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###All Oral Speeches
    oral = textData('ORAL').corpus
    features_dict = find_features(oral)
    #print(features_dict)
    with open('..\\data\\oral_train_scores.json', 'w') as f:
        json.dump(features_dict, f)


if __name__ == '__main__':
    main()
