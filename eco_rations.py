from nltk.corpus import inaugural
from nltk.corpus import stopwords

eco_rations = dict()
stop_words = set(stopwords.words('english'))
for fileid in inaugural.fileids():
    count = 0
    nonstw = 0
    for word in  inaugural.words(fileid):
        for target in ["economy","employment","export","unemploy","export"]:
            if not word.lower() in stop_words:
                nonstw=nonstw+1
                if word.lower().startswith(target):
                    count=count+1
    eco_rations[fileid[:4]] = count/nonstw
print(eco_rations)
                

