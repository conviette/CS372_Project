from nltk.corpus import reuters, stopwords, wordnet
from nltk import FreqDist
import re, math

keys = ['crude', 'fuel', 'gas', 'gnp', 'gold', 'income', 'interest', 'ipi', 'money-fx', 'money-supply', 'trade']
STOP = stopwords.words('english')
keywords = dict()
root = wordnet.synset('economy.n.01')

def scoring_func(word, freq):
    score = 0
    syns = wordnet.synsets(word)
    for s in syns:
        sim = root.path_similarity(s)
        if sim != None:
            score += sim
    return score*(freq**0.5)

for key in keys:
    print(key)
    fileids = reuters.fileids(key)
    corpus = [word.lower() for x in fileids for word in reuters.words(x)]
    corpus = list(filter(lambda x: (not x in STOP) and re.match('^\w+$', x) and len(x)>2, corpus))
    print(len(corpus))
    fd = FreqDist(corpus)
    score = dict()
    for word, freq in fd.most_common(60):
        score[word] = scoring_func(word, freq)
    keywords[key] = sorted(list(score.keys()), key=lambda x:score[x], reverse=True)[:30]
    print(keywords[key])

with open('reuters_keywords.json', 'w') as f:
    json.dump(keywords, f)
