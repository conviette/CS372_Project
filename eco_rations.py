from nltk.corpus import inaugural, stopwords, wordnet
from nltk.wsd import lesk


eco_rations = dict()
stop_words = set(stopwords.words('english'))


def syn_to_lem(key_synsets):
    words = set()
    for syns in key_synsets:
        words.update(list(map(lambda x:x.name(), syns.lemmas())))
    return words

def clean_corpus(corpus):
    return list(map(lambda sent:list(filter(lambda word:not word in stop_words, sent)), corpus))



def main():
    with open('economic_keywords.txt') as f:
        keywords = f.readlines()
    key_synsets = set(map(lambda x:wordnet.synset(x), keywords))
    keywords = syn_to_lem(key_synsets)



    for fileid in inaugural.fileids():
        corpus = clean_corpus(inaugural.sents(fileid))
        count = 0
        total_len = sum(map(len, corpus))
        for sent in corpus:
            check = set(filter(lambda x:x in sent, keywords))
            #check_synsets = set(map(lambda x:lesk(sent, x), check))
            count += len(check)
        eco_rations[fileid[:-4]] = count/total_len
    print(eco_rations)

    pres = list(eco_rations.keys())
    pres.sort(key=lambda x:eco_rations[x])
    print(pres)



if __name__ == '__main__':
    main()
