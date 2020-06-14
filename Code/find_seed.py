from nltk.corpus import brown
from nltk.corpus import reuters, stopwords
from nltk.probability import FreqDist
import re
from nltk.corpus import wordnet as wn

stop_words = set(stopwords.words('english'))
delimiters = {",",".","(",")","&","<",">",";",":"}
modals = {"would","will","said"}
month = {"january","february","march","april","may","june","july","august",
         "september","october","november","december"}
numbers = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
           "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
           "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
           "thousand", "million", "billion"}
time = {"day","days","week","weeks","years"}
stop_words.update(delimiters)
stop_words.update(modals)
stop_words.update(month)
stop_words.update(numbers)
stop_words.update(time)
global_text = reuters.words()
global_words = FreqDist(global_text).most_common(1000)
text = reuters.words(categories=["income","trade","earn","cpi","gnp","jobs","money-supply"])
ntext = list()
for word in text:
    w = word.lower()
    if (not w in stop_words) and (not w.isnumeric()) and (not w in global_words):
        ntext.append(w)
pre_seeds = FreqDist(ntext).most_common(100)
#print(pre_seeds)
seeds = set()
for word in pre_seeds:
#    print(word)
    for syn in wn.synsets(word[0]):
        seeds.add(syn)
print(seeds)
