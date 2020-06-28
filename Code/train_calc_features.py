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

def find_features_per_corpus(corpus_dict): #input corpus_dict: dict {year:[list of sents]}
    features_for_year = dict()
    keywords = set()
    for fileid in corpus_dict: #for each year
        features_for_year[fileid] = dict()
        corpus = clean_corpus(corpus_dict[fileid]) #clean corpus
        vocabulary_unigram = FreqDist([word for sent in corpus for word in sent])
        unigram_pdist = DictionaryProbDist(vocabulary_unigram, normalize=True)
        unigram_pdist = dict((x[0], unigram_pdist.prob(x[0])) for x in vocabulary_unigram.most_common(100))
        for key in unigram_pdist:
            keywords.add(key)
        features_for_year[fileid].update(unigram_pdist)
        big_corpus = [(s[i], s[i+1]) for s in corpus for i in range(len(s)-1)]
        vocab_bigram = FreqDist(big_corpus)
        bigram_pdist = DictionaryProbDist(vocab_bigram, normalize=True)
        bigram_pdist = dict(('_'.join(x[0]), bigram_pdist.prob(x[0])) for x in vocab_bigram.most_common(30))
        for key in bigram_pdist:
            keywords.add(key)
        features_for_year[fileid].update(bigram_pdist)
        #add bigrams


    for d in features_for_year.values():
        for key in keywords:
            if not key in d:
                d[key] = 0.0

    return features_for_year #output features_for_year: dict {year:{feature_name:float}}

def find_features_agg(corpus_dict):
    features_for_year = dict()
    keywords = set()
    supercorpus = []
    for fileid in corpus_dict: #for each year
        features_for_year[fileid] = dict()
        supercorpus.extend(clean_corpus(corpus_dict[fileid])) #clean corpus
    super_unigram = [x[0] for x in FreqDist([word for sent in supercorpus for word in sent]).most_common(1500)]
    super_bigram = ['_'.join(x[0]) for x in FreqDist([(s[i], s[i+1]) for s in supercorpus for i in range(len(s)-1)]).most_common(500)]

    for fileid in corpus_dict:
        corpus = clean_corpus(corpus_dict[fileid])
        vocabulary_unigram = FreqDist([word for sent in corpus for word in sent])
        unigram_pdist = DictionaryProbDist(vocabulary_unigram, normalize=True)
        for key in super_unigram:
            if key in vocabulary_unigram:
                features_for_year[fileid][key] = unigram_pdist.prob(key)
            else:
                features_for_year[fileid][key] = 0.0

        big_corpus = [s[i]+'_'+s[i+1] for s in corpus for i in range(len(s)-1)]
        vocab_bigram = FreqDist(big_corpus)
        bigram_pdist = DictionaryProbDist(vocab_bigram, normalize=True)
        for key in super_bigram:
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
    with open('..\\data\\inaugural_train_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###State of the Union Speech
    sotus = textData('SOTUS').corpus
    features_dict = find_features_agg(sotus)
    #print(features_dict)
    with open('..\\data\\sotus_train_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###All Oral Speeches
    oral = textData('ORAL').corpus
    features_dict = find_features_agg(oral)
    #print(features_dict)
    with open('..\\data\\oral_train_scores.json', 'w') as f:
        json.dump(features_dict, f)


if __name__ == '__main__':
    main()