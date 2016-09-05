import sklearn
import pandas
import csv
import string
import nltk
from pprint import pprint
from nltk import tokenize
import re
import numpy as np
file2=open("results.csv",mode="w")
texts=[]
results=[]
file3=open("surprise_out.txt","w",encoding="utf-8")
from sklearn import cross_validation
from sklearn import dummy
from sklearn import feature_extraction
from sklearn import grid_search
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm
texts=[]
with open('surprise-hindi.csv',encoding="utf-8",errors="ignore") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        texts.append(row["tweet_text"])
#texts=["मेरी ड्राइविंग पसंद नहीं है।"]
pprint(texts)
'''vectorizer = feature_extraction.text.TfidfVectorizer()
unigrams = vectorizer.fit_transform(texts).toarray()'''
#pprint(unigrams)
file=pandas.read_csv("nrc-hindi1.csv",error_bad_lines=False,encoding="utf-8")
#file1=pandas.read_csv("anger-hindi1.csv",error_bad_lines=False,encoding="utf-8")
c=np.array([0,0,0,0,0,0,0,0,0,0])

#print(file['Hindi Translation (Google Translate)'][2])
for sent in texts:
    num = 0
    t1 = -1
    trans = 1
    index_conj = -1
    index_n = -1
    index_not = -1
    sent = " " + sent
    index_c = -1
    sent = sent.replace(",", " और ")
    print(sent)
    c=np.array([[0,0,0,0,0,0,0,0,0,0]])
    #out = sent.translate(sent.maketrans("", "", string.punctuation))

    #print(file[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']])
    index_b = -1
    list_of_con = [" और ", " चाहे ", " या ", " अगर ", " तो ", " क्यूंकि ", " चूँकि ", " एंव ", " इसलिए ", " जैसे "]
    for conj in list_of_con:
        sent = sent.replace(conj, " और ")
    list_of_buts = [" लेकिन ", " मगर ", " किन्तु ", " फिर भी ", " परंतु ", " पर "]
    list_of_in = [" अति ", " ऊँची ", " अत्यधिक ", " ऊंचा ", " महान ", " काफ़ी "]
    for intensifier in list_of_in:
        sent = sent.replace(intensifier, " बहुत ")
    list_of_dim = [" हल्का ", " सामान्य ", " नीचा ", " निची ", " कम ", " थोड़ा "]
    for dim in list_of_dim:
        sent = sent.replace(dim, " नीच ")
    list_of_words = nltk.word_tokenize(sent)
    pprint(list_of_words)
    sent_rev = ""
    for word in reversed(list_of_words):
        sent_rev = sent_rev + word + " "
    print(sent_rev)
    list = []
    for b in list_of_buts:
        if sent_rev.find(b) > 0:
            list.append(sent_rev.find(b))
    pprint(list)
    list.sort()
    if len(list) != 0:
        index_b = list[0]
    while index_b != -1:
        sent_rev = sent_rev[:index_b]
        for b in list_of_buts:
            index_b = sent_rev.find(b)
    print(sent_rev)
    list_of_words = nltk.word_tokenize(sent_rev)
    pprint(list_of_words)
    sent_rev = ""
    for word in reversed(list_of_words):
        sent_rev = sent_rev + word + " "
    sent = sent_rev
    sent_list = sent.split(" और ")
    sent1 = " "
    for s in sent_list:
        index_not = s.find(" नहीं ") + 5
        print(index_not)
        print(s[index_not:])
        index_conj = s[index_not:].find(" कि ") + index_not
        print(index_conj)
        if index_conj >= index_not and index_not > 4:
            print(s[index_conj:])
            s = s[index_conj:] + s[index_not - 5:index_not] + s[:index_not - 4] + s[index_not:index_conj]
            sent1 = sent1 + s + " "
        else:
            sent1 = sent
    sent = sent1
    sent = " " + sent
    print(sent)
    index_n = sent.find(" न ") + 4
    print(index_n)
    index_c = -1
    while index_n > 3:
        index_c = sent[index_n:].find("और ") + index_n
        print(index_c)
        if index_c > index_n:
            print(index_c)
            sent = sent[:index_n - 3] + sent[index_n:index_c] + " नहीं " + sent[index_c + len("और"):]
            print(sent[index_c+5:])

        else:
            sent = sent[:index_n - 3] + sent[index_n:] + " नहीं "
        index_n = sent.find(" न ") + 4
    print(sent)
    result=np.array([0,0,0,0,0,0,0,0,0,0])

    d = nltk.word_tokenize(sent)
    for k in d:
        print(d[num],num)

        if d[num] == "नहीं":
            if d[num - 1] != "":
                c = -c
        for i in range(0, file.count()[1]):
            #c = np.array([0,0,0,0,0,0,0,0,0,0])
            lexword = file['Hindi Translation (Google Translate)'][i]
            if k == lexword:
                print("helllo")
                score=file[['Positive','Negative','Anger','Anticipation','Disgust','Fear','Joy','Sadness','Surprise','Trust']][i:i+1].values[0].astype(float)
                # = file[['Positive', 'Negative', 'Disgust']][i:i + 1].values[0].astype(float)
                if d[num - 1] == "बहुत":
                    score = 1.5 * score
                elif d[num - 1] == "नीच":
                    score = 0.5 * score
                pprint(score)
                print("ehllo")
                #pprint(c)
                c = c + score
                break
        num = num + 1
        pprint(c)
    if len(d)>0:
        result = c / len(d)
    result=result.tolist()[0]
    if type(result)=="list":
        if result[0] < 0.:
            result[1] = result[1] - result[0]
            result[0] = 0
        if result[1] < 0.:
            result[0] = result[0] - result[1]
            result[1] = 0
        for i in range(2, len(result)):
            if (result[i] <= 0.):
                result[i] = 0.0
    pprint(result)
    file3.write(sent+", "+str(result)+"\n")