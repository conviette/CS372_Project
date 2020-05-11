from nltk.corpus import wordnet as wn

def convert_pos(syn_name):
    #find similar words but have different pos
    # Get all lemmas
    lemmas = [l for l in wn.synset(syn_name).lemmas()]

    # get related words
    related = [r for l in lemmas for r in l.derivationally_related_forms()]
    related += [r for l in lemmas for r in l.pertainyms()]

    # Extract the synsets from the lemmas
    new_synsets = [l.synset() for l in related]

    return new_synsets

seed = ['economy.n.01', 'industry.n.01','work.n.01','trade.n.01',  'worker.n.01', 'interest.n.04', 'money.n.01', 'market.n.01',
         'opportunity.n.01', 'success.n.03', 'government.n.01']
seed = list(map(lambda x:wn.synset(x), seed))
group = []



for s in seed:
    lexicon = set([s])
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s.closure(hypos)))
    for synset in convert_pos(s.name()):
        lexicon.update(set(synset.closure(hypos)))



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