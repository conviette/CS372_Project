import nltk
from nltk.corpus import *

def other_word_similarity(word, selected):
    def find_similarity(synset):
        best = 0
        for s in selected:
            sim = synset.path_similarity(s)
            if sim and sim>best:
                best = sim
        return best

    best = 0
    for s in wordnet.synsets(word):
        sim = find_similarity(s)
        if sim>best: best = sim

    return best

words = brown.words(categories = 'news')

fdist = nltk.FreqDist(words)
x = fdist.most_common()

manual = ['economy', 'crisis', 'power', 'worker', 'resource']
ans = []

for word in manual:
    selected = []
    for s in wordnet.synsets(word):
        selected.append(s)
    cnt = 0
    for (w, size) in x:
        sim = other_word_similarity(w, selected)
        if sim > 0.2:
            print(word, w, sim)
            ans.append(w)
            cnt += 1
            if cnt == 10:
                break




print(ans)