import json, re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import inaugural, stopwords
from collections import defaultdict

class textData:
    def __init__(self, name):
        assert name in ('INAUGURAL', 'SOTUS', 'ORAL')
        if name == 'INAUGURAL':
            self.corpus = self._getInaug()
        if name == 'SOTUS':
            self.corpus = self._getSOTUS()
        if name == 'ORAL':
            self.corpus = self._getORAL()

    def _process_bodytext(self, text): #for text that is not tokenized
        text = sent_tokenize(text)
        return list(map(word_tokenize, text))


    def _getInaug(self):
        return dict((fileid[:4], inaugural.sents(fileid)) for fileid in filter(lambda x:int(x[:4])>=1960, inaugural.fileids()))

    def _getSOTUS(self):
        with open('..\\data\\Presidential\\sotus_a.json') as f:
            sotus = json.load(f)
        return dict((x['date'][-4:], self._process_bodytext(x['body'])) for x in sotus)

    def _getORAL(self):
        with open('..\\data\\Presidential\\oral_a.json') as f:
            oral = json.load(f)
        corpus = defaultdict(list)
        for x in oral:
            body = self._process_bodytext(x['body'])
            corpus[x['date'][-4:]].extend(body)
        return corpus
