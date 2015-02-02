import csv
import nltk
import re
import pandas as pd
import MultiClass
import warnings

df = pd.read_excel("E:/TripAdvisor.xlsx")
reviews = list(df["Review"])
#print(reviews)

counter = 0
cat = []

multiClassifier = MultiClass.MultiClassiflier()

multiClassifier.predict(reviews)

warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)

