import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import warnings

class MultiClassiflier:
    def predict(self,reviews):
        dir = os.path.dirname(__file__)
        tt = pd.read_csv(dir + '/Data/TrainingSet/Data.csv', sep=",")
        arrayData = [var for var in tt["name"] if var]
        X_train = np.array(arrayData)
        #print(X_train)
        #print(len(arrayData))
        #y_train = [[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0,1],[0,1]]
        y_train = []
        categoryData = [var for var in tt["category"] if var]
        for ids in categoryData:
            indexarry=[]
            splitIds = str(ids).split(',')
            for id in splitIds:
                indexarry.append(int(id))
            #print(indexarry)
            y_train.append(indexarry)
        X_test = np.array(reviews)
        catData = pd.read_csv(dir + '/Data/TrainingSet/Categories.csv', sep=",")        
        target_names = list(catData["CategoryName"])
        #target_names = ['Food', 'Fruit','Nuts','Vegetable','Desert','Soup']

        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LinearSVC()))])
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        for item, labels in zip(X_test, predicted):
            print('%s => %s' % (item, ', '.join(target_names[x] for x in labels)))


