from nltk.corpus import wordnet as wn
import json
#I replaced 'work.n.01' by its hyponyms related to economy
seed = ['economy.n.01', 'trade.n.01', 'market.n.01', 'money.n.01', 'job.n.06', 'service.n.01', 'gold.n.05']
group = dict()

for s in seed:
    s_syn = wn.synset(s)
    print(s_syn.name())
    lexicon = set([s_syn])
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s_syn.closure(hypos)))
    group[s] = list(map(lambda x:x.name(), lexicon))

with open('..\\data\\keyword_group.json', 'w') as f:
    json.dump(group, f)


'''
Something to do more
add verbs, adjectives to each group.
the group of economy do not contain 'economic' and 'economic' appears 47 times for all speeches.

some words are missing: ex) worker, ...

remove unrelated words: ex) x-ray_therapy.n.01
'''
