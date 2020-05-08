from nltk.corpus import wordnet as wn

seed = ['economy.n.01', 'trade.n.01', 'market.n.01', 'money.n.01', 'work.n.01', 'gold.n.05']
seed = list(map(lambda x:wn.synset(x), seed))
lexicon = set(seed)

for s in seed:
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s.closure(hypos)))

with open('economic_keywords.txt', 'w') as f:
    for l in lexicon:
        f.write(l.name()+'\n')
