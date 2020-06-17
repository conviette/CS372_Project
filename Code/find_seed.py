from nltk.corpus import reuters, stopwords
from nltk.probability import FreqDist
import re
from nltk.corpus import wordnet as wn
import math, json

targets = ["income","trade","earn","cpi","gnp","jobs","money-supply"]
non_targets = [cat for cat in reuters.categories() if cat not in targets]
stop_words = set(stopwords.words('english'))
delimiters = {",",".","(",")","&","<",">",";",":","\""}
modals = {"would","will","said"}
months = {"january","february","march","april","may","june","july","august",
         "september","october","november","december"}
days = {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"}
numbers = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
           "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
           "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
           "thousand", "million", "billion"}
time = {"day","days","week","weeks","year","years"}
stop_words.update(delimiters)
stop_words.update(modals)
stop_words.update(months)
stop_words.update(days)
stop_words.update(numbers)
stop_words.update(time)

dfs = dict()
num_docs = 0
for fid in reuters.fileids(non_targets):
    num_docs = num_docs + 1
    words = set([word.lower() for word in reuters.words(fileids = fid)])
    for word in words:
        if word in dfs:
            dfs[word] = dfs[word] + 1
        else:
            dfs[word] = 1
counts = dict()
doc_length = dict()
for cat in targets:
    num_docs = num_docs + 1
    counts[cat] = dict()
    words = set([word.lower() for word in reuters.words(categories = cat)])
    for word in words:
        if word in dfs:
            dfs[word] = dfs[word] + 1
        else:
            dfs[word] = 1
    for word in reuters.words(categories = cat):
        w = word.lower()
        if w in counts[cat]:
            counts[cat][w] = counts[cat][w] + 1
        else:
            counts[cat][w] = 1
    doc_length[cat] = len(reuters.words(categories = cat))

tf_idf = dict()
for cat in targets:
    tf_idf[cat] = dict()
    for word in counts[cat]:
        tf = counts[cat][word]/doc_length[cat]
        idf = math.log(num_docs/dfs[word])
        tf_idf[cat][word] = tf * idf
data = dict()
for cat in tf_idf:
    print(cat)
    words = list()
    answers = sorted(tf_idf[cat].items(), key = lambda x: x[1], reverse=True)
    for answer in answers:
        if (not answer[0] in stop_words) and (not answer[0].isnumeric()):
            words.append(answer[0])
    print(words[:60])
    data[cat] = words[:60]

with open('keywords_by_tf_idf.json', 'w') as f:
    json.dump(data, f)


