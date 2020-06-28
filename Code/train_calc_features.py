from nltk.corpus import inaugural, stopwords, wordnet
from nltk import word_tokenize, sent_tokenize, FreqDist, DictionaryProbDist
import numpy as np
import json, re
from project_dataset import textData
'''
    added
'''
import feature_tense

##calculating features for each year

##get stopwords
stop_words = set(stopwords.words('english'))

def clean_corpus(corpus): #any preprocessing for corpus
    corpus = list(map(lambda sent:list(filter(lambda word:(not word in stop_words) and len(word)>3, sent)), corpus))
    return [[word.lower() for word in sent] for sent in corpus]#discarding stopwords

#####functions for computing each features#########
#####all functions must accept feature_dict as input, and add entry "feature_name" mapped to number


def find_features_agg(corpus_dict): ###edit this function to add featuress
    features_for_year = dict()
    keywords = set()
    supercorpus = []
    for fileid in corpus_dict: #for each year
        features_for_year[fileid] = dict()
        supercorpus.extend(clean_corpus(corpus_dict[fileid])) #concatenated corpus (all)

    ##list of possible features (ex for unigrams, list of 1500 most common unigrams in all of corpus)
    super_unigram = [x[0] for x in FreqDist([word for sent in supercorpus for word in sent]).most_common(1500)]
    super_bigram = ['_'.join(x[0]) for x in FreqDist([(s[i], s[i+1]) for s in supercorpus for i in range(len(s)-1)]).most_common(500)]
    '''
    added
    '''
    temp = []
    for sent in supercorpus:
        if not sent: continue
        time,pn = feature_tense.tense(' '.join(sent))
        for word in sent:
            temp.append(word + '_' + time + '_' +pn)

    temp = FreqDist(temp).most_common(500)
    super_tense = [t[0] for t in temp]
    print(super_tense)



    for fileid in corpus_dict:
        corpus = clean_corpus(corpus_dict[fileid])

        ####computing feature for EACH document###
        vocabulary_unigram = FreqDist([word for sent in corpus for word in sent])
        unigram_pdist = DictionaryProbDist(vocabulary_unigram, normalize=True)
        for key in super_unigram:
            if key in vocabulary_unigram: ##fileid = document id, key=unigram
                features_for_year[fileid][key] = unigram_pdist.prob(key) ###normalized frequency of unigram for document
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

        ####add feature here#####
        ####save frequency to features_for_year[fileid][featurename]
        ####all documents should have same features (no features that exist in only some of the documents)

        '''
            added
        '''
        tense_word = []
        for sent in corpus:
            if not sent: continue
            time,pn = feature_tense.tense(' '.join(sent)) #tense and positive/negative
            for word in sent:
                tense_word.append(word + '_' + time + '_' +pn)
        tense_fdist = FreqDist(tense_word)
        tense_pdist = DictionaryProbDist(tense_fdist, normalize=True)
        for key in super_tense:
            if key in tense_fdist:
                features_for_year[fileid][key] = tense_pdist.prob(key)
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
