from nltk.corpus import inaugural, stopwords, wordnet, conll2000
import nltk
from nltk.wsd import lesk
import numpy as np
from Linear_Regression import *
import matplotlib.pyplot as plt

'''
How can I use it on cal_features.py?
'''

class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)] for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        if not sentence: return None
        (words, pos) = zip(*sentence)
        chunks = self.tagger.tag(pos)
        chunk_sent = [(w, p, c) for (w, (p, c)) in zip(words, chunks)]
        return nltk.chunk.conlltags2tree(chunk_sent)

train_txt = conll2000.chunked_sents('train.txt')
chunker = BigramChunker(train_txt)

def _find_node(word, position, tree):
    cnt = 0
    for i in range(len(tree)):
        phrase = tree[i]
        if type(phrase) == type((1,2)): #tuple
            if phrase[0] == word:
                cnt += 1
        else:
            for w,p in phrase:
                if w == word:
                    cnt += 1

        if cnt >= position: return i

def _nearest_verb(tree, idx):
    def phrase_type(data):
        '''
        determine the label if type of data is tree or pos if type of data is tuple

        If given data is chunk, it will return name of chunk
        If given data is not chunk, it will return pos

        :param tree: nltk.tree
        :return: string
        '''
        pos = data[-1]
        if type(pos) == type('a'):
            return pos
        else:
            return data.label()

    targ_phrase = tree[idx]
    if phrase_type(targ_phrase) == 'VP':
        return idx
    elif phrase_type(targ_phrase) == 'NP' or phrase_type(targ_phrase).startswith('R'):
        a, b = 0, 0
        tree_size = len(tree)
        while idx + a < tree_size and phrase_type(tree[idx+a]) != 'VP':
            a+=1
        while idx-b >= 0 and phrase_type(tree[idx-b]) != 'VP':
            b+=1

        if idx + a == tree_size:
            if idx - b >= 0: return idx-b
        else:
            if idx - b == -1: return idx + a
            else:
                if a<b: return idx + a
                else: return idx - b

    return -1

def _score_verb(tree,idx):
    time = 0  # -1 for past, 0 for present, 1 for future
    pos = 1  #1 for positive, -1 for negative
    VP = tree[idx]

    #check future or past
    if type(VP) == type((1,2)): return time,pos
    for i in range(len(VP)):
        w,p = VP[i]
        if p == 'VBD' or p == 'VBN':
            time = -1
            break
        if w == 'will':
            time = 1
            break
        if w == 'to' and i>1:
            if VP[i-1][0] == 'going':
                if VP[i-2][0] != 'not' and VP[i-2][0] != "n't":
                    if wordnet.morphy(VP[i-2][0],wordnet.VERB) == 'be':
                        time = 1
                        break
                else:
                    if i>2 and wordnet.morphy(VP[i-3][0],wordnet.VERB) == 'be':
                        time = 1
                        break

    #check negative
    for w,p in VP:
        if w == 'not' or w == "n't":
            pos = -1

    return time, pos

def parse(time,pos):
    t = 'past'
    if time == 0: t = 'present'
    elif time == 1: t = 'future'

    pn = 'pos'
    if pos == -1: pn = 'neg'

    return t,pn

def tense(sent, word=None):
    '''
    get tense of target word(+ pos/neg)

    You can only analysis with only sentence, or analysis with a sentence and a word in the sentence
    The latter can give more reliable output

    return: time(arg1):     -1/0/1  => past/present/future tense
            pos(arg2):      -1/1    => neg/pos

    :param sent:    string, not tokenized
    :param word:    string, target word to be analyzed
    :return:        (int, int), tense, pos/neg
    '''

    token = nltk.word_tokenize(sent)
    tag = nltk.pos_tag(token)
    tree = chunker.parse(tag)

    node_idx = 0
    if word:
        node_idx = _find_node(word, 1, tree)
    verb_idx = _nearest_verb(tree,node_idx)
    time,sent = _score_verb(tree,verb_idx)

    return parse(time,sent)



#print(tense("I don't have a car"))
