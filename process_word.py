from nltk.corpus import inaugural, stopwords, wordnet
from nltk.wsd import lesk
import numpy as np
from Linear_Regression import *
import matplotlib.pyplot as plt

'''
compatible with find_keywords_group.py and Linear_Regression.py
this finds # of seed words and accuracy
'''

eco_rations = dict()
stop_words = set(stopwords.words('english'))


def syn_to_lem(key_synsets):
    words = set()
    for syns in key_synsets:
        s = wordnet.synset(syns)
        words.update(list(s.lemma_names()))
    return words

def clean_corpus(corpus):
    return list(map(lambda sent:list(filter(lambda word:not word in stop_words, sent)), corpus))



def concat_all(data):  # concatinate all array in a list, all elements should have same number of columns
    dat = data[0]
    for ind in range(1, len(data)):
        dat = np.concatenate((dat, data[ind]), axis=0)
    return dat

def main():
    synset_group = []
    word_group = []
    president_Synset_usage = []
    f = open("keyword_group.txt", 'r')
    while True:
        synset = f.readline()
        if not synset: break
        synset = synset.strip()
        if synset == 'group':
            synset_group.append(set())
        else:
            synset_group[-1].add(synset)
    f.close()

    group_size = len(synset_group)
    for g in synset_group:
        word_group.append(syn_to_lem(g))

    for fileid in inaugural.fileids():
        corpus = clean_corpus(inaugural.sents(fileid))
        total_len = sum(map(len, corpus))

        Group_usage = np.zeros((1, group_size))
        for sent in corpus:
            for i in range(group_size):
                keywords = word_group[i]
                check = set(filter(lambda x: x in sent, keywords))
                Group_usage[0][i] += len(check)

        Group_usage /= total_len

        president_Synset_usage.append((fileid[:-4],Group_usage))

    #for f in president_Synset_usage:   print(f)

    learnable = []
    for f in president_Synset_usage[-15:]:
        learnable.append(f[1])

    data = concat_all(learnable)
    label = np.array([[4.65,5.05,2.86,2.58,3.24, 3.14,3.82, 2.25, 3.31, 4.45,2.35,2.03,1.46,2.19,2.48]])
    label = label.T
    #best_lambda(3,data,label)
    train_acc = []
    l_list = []

    idx = np.arange(group_size)
    np.random.seed(6)
    np.random.shuffle(idx)
    for i in range(group_size):
        dt = data.T
        #dt = dt[idx[:i+1]]
        dt = dt[:i+1]
        #train, test = k_fold(3,dt.T,label)
        train,test,l = best_lambda(3, dt.T, label)
        train_acc.append(train)
        l_list.append(l)

    print('\nl: ', l_list)
    print('\ntrain: ',train_acc)

    x_range = np.arange(group_size)
    plt.plot(x_range,train_acc,c='k')
    plt.xlabel('number of seed words')
    plt.ylabel('R2 loss')
    plt.show()




if __name__ == '__main__':
    main()
