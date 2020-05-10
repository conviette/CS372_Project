from nltk.corpus import wordnet as wn
import nltk

def other_pos_word(syn_name):
    # 
    # Get all lemmas
    word = syn_name.split('.')[0]
    word_pos = syn_name.split('.')[1]
    synsets = wn.synsets(word, pos=word_pos)
    if (word_pos == 'a'): #both 'a' and 's' are adjective
        synsets += wn.synsets(word, pos='s')
    if not synsets:
        return 0
    lemmas = [l for s in synsets for l in s.lemmas()]

    # get related words
    related = [related for l in lemmas for related in l.derivationally_related_forms()]


    # Extract the words from the lemmas
    words = [l.name() for l in related]
    word_freq = nltk.FreqDist(words)
    word_len = len(words)
    word_usage = [(word_freq[w] / word_len, w) for w in set(words)]

    #get synsets to get similarity
    changed_words_synsets = [wn.synsets(s) for p,s in word_usage if p>0.1]
    res = [s for syns in changed_words_synsets for s in syns]

    return res


seed = ['economy.n.01', 'trade.n.01', 'market.n.01', 'money.n.01', 'work.n.01', 'gold.n.05']
other_pos = [other_pos_word(syn_name) for syn_name in seed]
seed = list(map(lambda x:wn.synset(x), seed))
seed += [s for syns in other_pos for s in syns]
lexicon = set(seed)



for s in seed:
    hypos = lambda x:x.hyponyms()
    lexicon.update(set(s.closure(hypos)))

with open('economic_keywords.txt', 'w') as f:
    for l in lexicon:
        f.write(l.name()+'\n')
