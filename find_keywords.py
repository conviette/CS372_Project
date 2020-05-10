from nltk.corpus import wordnet as wn
import nltk

def convert_and_similarity(syn_name):
    # Get all lemmas
    lemmas = [l for l in wn.synset(syn_name).lemmas()]

    # get related words
    related = [r for l in lemmas for r in l.derivationally_related_forms()]
    related += [r for l in lemmas for r in l.pertainyms()]

    # Extract the synsets from the lemmas
    new_synsets = [l.synset() for l in related]

    return new_synsets


seed = ['economy.n.01', 'trade.n.01', 'market.n.01', 'money.n.01', 'work.n.01', 'worker.n.01']
other_pos = [convert_and_similarity(syn_name) for syn_name in seed]
print(other_pos)
seed = list(map(lambda x:wn.synset(x), seed))
seed += [s for syns in other_pos for s in syns]
lexicon = set(seed)



for s in seed:
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s.closure(hypos)))

with open('economic_keywords.txt', 'w') as f:
    for l in lexicon:
        f.write(l.name()+'\n')
