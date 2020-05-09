from nltk.corpus import wordnet as wn

seed = ['economy.n.01', 'trade.n.01', 'market.n.01', 'money.n.01', 'work.n.01', 'gold.n.05']
seed = list(map(lambda x:wn.synset(x), seed))
group = []

for s in seed:
    lexicon = set([s])
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s.closure(hypos)))
    group.append(lexicon)

with open('keyword_group.txt', 'w') as f:
    for g in group:
        f.write('group\n')
        for s in g:
            f.write(s.name()+'\n')


'''
Something to do more
add verbs, adjectives to each group.
the group of economy do not contain 'economic' and 'economic' appears 47 times for all speeches.

some words are missing: ex) worker, ...

remove unrelated words: ex) x-ray_therapy.n.01
'''