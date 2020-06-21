from nltk.corpus import inaugural, stopwords, wordnet
from nltk import word_tokenize, sent_tokenize
import numpy as np
import json
from project_dataset import textData

##calculating features for each year

##get stopwords
stop_words = set(stopwords.words('english'))


###get keyword dictionary###
def syn_to_lem(key_synsets):
    words = set()
    for syns in key_synsets:
        syns = wordnet.synset(syns)
        words.update(list(map(lambda x:' '.join(x.name().split('_')).lower(), syns.lemmas())))
    return words

def get_keywords_wordnet(): #from wordnet
    with open('..\\data\\keyword_group.json') as f:
        keywords = json.load(f)
    keywords = dict((x.split('.')[0], syn_to_lem(keywords[x])) for x in keywords.keys())
    key_synsets = list(keywords.keys())
    return keywords, key_synsets

def lemmatize_group(l):
    output = set()
    for word in l:
        synsets = wordnet.synsets(word)
        for syns in synsets:
            output.update(list(map(lambda x:' '.join(x.name().split('_')).lower(), syns.lemmas())))
    return list(output)

def get_keywords_reuters():
    with open('..\\data\\reuters_keywords.json') as f:
        keywords = json.load(f)
    keywords = dict((x, lemmatize_group(keywords[x])) for x in keywords)
    key_synsets = list(keywords.keys())
    return keywords, key_synsets

def get_keywords_tf_idf():
    with open('..\\data\\keywords_by_tf_idf.json') as f:
        keywords = json.load(f)
    keywords = dict((x, lemmatize_group(keywords[x])) for x in keywords)
    key_synsets = list(keywords.keys())
    return keywords, key_synsets
        

##keywords = dictionary, {feature name:[list of keywords (with all forms of lemmatization)]}
##ex. {'money':['money', 'monetize', 'currency', 'currencies'...]}
##key_synsets = list of feature names (ex. ['money', 'trade', 'economy'...])

keywords, key_synsets = get_keywords_wordnet() #or get_keyword_reuters, or any other keyword getting function

def clean_corpus(corpus): #any preprocessing for corpus
    return list(map(lambda sent:list(filter(lambda word:not word in stop_words, sent)), corpus)) #discarding stopwords

#####functions for computing each features#########
#####all functions must accept feature_dict as input, and add entry "feature_name" mapped to number
def count_frequencies(corpus, feature_dict, total_len):
    for key in keywords:
        feature_dict[key] = 0
    for sent in corpus:
        for key, pool in keywords.items():
            feature_dict[key] +=count_from_pool(sent, pool)/total_len

def count_from_pool(sent, pool):
    sent = ' '.join(sent).lower()
    count = 0
    for w in pool:
        if w in sent:
            count+=1
    return count

def freq_dist(corpus, feature_dict, total_len):
    for key in ['first', 'middle', 'last']:
        feature_dict[key] = 0
    first, middle, last = corpus[:len(corpus)//3], corpus[len(corpus)//3:len(corpus)*2//3], corpus[len(corpus)*2//3:]
    keyword_pool = []
    for key, pool in keywords.items():
        keyword_pool.extend(pool)
    for sent in first:
        feature_dict['first'] +=count_from_pool(sent, keyword_pool)/total_len
    for sent in middle:
        feature_dict['middle'] +=count_from_pool(sent, keyword_pool)/total_len
    for sent in last:
        feature_dict['last'] +=count_from_pool(sent, keyword_pool)/total_len


def find_features(corpus_dict): #input corpus_dict: dict {year:[list of sents]}
    features_for_year = dict()
    for fileid in corpus_dict: #for each year
        corpus = clean_corpus(corpus_dict[fileid]) #clean corpus
        total_len = sum(map(len, corpus)) #total length of corpus (to normalize)
        feature_dict = dict() #dictionary: map feature name->feature(number)
        #######calculate features#######
        count_frequencies(corpus, feature_dict, total_len)
        #freq_dist(corpus, feature_dict, total_len)

        features_for_year[fileid] = feature_dict
    return features_for_year #output features_for_year: dict {year:{feature_name:float}}



def main():

    ###Inaugural corpus
    inaug = textData('INAUGURAL').corpus
    features_dict = find_features(inaug) #The word the President said and its amount
    #print(features_dict)
    with open('..\\data\\inaugural_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###State of the Union Speech
    sotus = textData('SOTUS').corpus
    features_dict = find_features(sotus)
    #print(features_dict)
    with open('..\\data\\sotus_scores.json', 'w') as f:
        json.dump(features_dict, f)

    ###All Oral Speeches
    oral = textData('ORAL').corpus
    features_dict = find_features(oral)
    #print(features_dict)
    with open('..\\data\\oral_scores.json', 'w') as f:
        json.dump(features_dict, f)


if __name__ == '__main__':
    main()
