import numpy as np

import pandas

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
from pprint import pprint
import csv
texts=[]

def indent(lines, amount, ch=' '):
    #http://stackoverflow.com/questions/8234274/how-to-indent-the-content-of-a-string-in-python
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))

with open('Book2.csv',encoding="utf-8",errors="ignore") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        texts.append(row["Text"])
data=pandas.read_csv("Book3.csv",encoding="utf-8")
pprint(data)
sentiment_scaler = preprocessing.StandardScaler()

models = [
          naive_bayes.GaussianNB(),
          linear_model.LogisticRegression(random_state=0),
          svm.SVC(random_state=0, kernel='linear'),
         ]

clf_hyp = [
           dict(),
           dict(clf__C=[.00001, .0001, .001, .01, .1, 1., 10.]),
           dict(clf__C=[.00001, .0001, .001, .01, .1, 1., 10.]),
          ]
#pprint(file.ix[:,'Positive':'Trust'])
'''sentiment = sentiment_scaler.fit_transform(file.ix[:,:'Trust'])
sentiment_t=sentiment_scaler.fit_transform(file.ix[:,'Positive_label':'Trust_label'])
pprint(sentiment)
vectorizer = feature_extraction.text.TfidfVectorizer()
pprint(texts)
unigrams = vectorizer.fit_transform(texts).toarray()
pprint(unigrams)
runBaseline = True'''


#trainX, testX, yTrain, yTest = cross_validation.train_test_split(X,Y,test_size=0.1, random_state=0)





X = data.iloc[:,10:]

for y in range(11,16):

    #print(indent("Personality Factor: ", 2), data.columns.values[y])
    Y = data.iloc[:,y]
    runBaseline = True

    trainX, testX, yTrain, yTest = cross_validation.train_test_split(X,Y,test_size=0.1, random_state=0)

    vectorizer = feature_extraction.text.TfidfVectorizer()
    sentiment_scaler = preprocessing.StandardScaler()
    pprint("hello")
    pprint(list(data.columns.values))
    unigrams = vectorizer.fit_transform(texts).toarray()

    sentiment = sentiment_scaler.fit_transform(trainX.ix[:,1:10])
    allf = np.hstack((unigrams, sentiment))

    unigrams_t = vectorizer.transform(texts).toarray()
#liwc_t = liwc_scaler.transform(testX.ix[:,"WC":"OtherP"])
    sentiment_t = sentiment_scaler.transform(testX.ix[:,11:16])
#pos_t = pos_scaler.transform(testX.ix[:, "''":"WRB"])
    allf_t = np.hstack((unigrams_t, sentiment_t))
    features = {"All":(allf, allf_t)}


    for f in features:
        xTrain = features[f][0]
        xTest = features[f][1]

        if runBaseline:
            baseline = dummy.DummyClassifier(strategy='most_frequent', random_state=0)
            baseline.fit(xTrain, yTrain)
            predictions = baseline.predict(xTest)

            print(indent("Baseline: ", 4))
            print(indent("Test Accuracy: ", 4), metrics.accuracy_score(yTest, predictions))
            print(indent(metrics.classification_report(yTest, predictions), 4))
            print()
            runBaseline = False

        print(indent("Features: ", 4), f)

        for m, model in enumerate(models):
            hyp = clf_hyp[m]
            pipe = pipeline.Pipeline([('clf', model)])

            if len(hyp) > 0:
                grid = grid_search.GridSearchCV(pipe, hyp, cv=10, n_jobs=-1)
                grid.fit(xTrain, yTrain)
                predictions = grid.predict(xTest)

                print(indent(type(model).__name__, 6))
                print(indent("Best hyperparameters: ", 8), grid.best_params_)
                print(indent("Validation Accuracy: ", 8), grid.best_score_)
                print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
                print(indent(metrics.classification_report(yTest, predictions), 8))

            else:
                grid = model
                grid.fit(xTrain, yTrain)
                predictions = grid.predict(xTest)

                print(indent(type(model).__name__, 6))
                print(indent("Test Accuracy: ", 8), metrics.accuracy_score(yTest, predictions))
                print(indent(metrics.classification_report(yTest, predictions), 8))

        print()
    print()