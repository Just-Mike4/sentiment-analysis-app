import numpy as np
import pandas as pd
import pickle
data=pd.read_csv(r"C:\Users\USER\Desktop\PY app\movie.csv")
data=data[:5000]
data.head()
column_name=['Review','Sentiment']
data.columns=column_name
data['Sentiment'].value_counts()
x=data["Review"]
y=data["Sentiment"]
import string
punc=string.punctuation
from spacy.lang.en.stop_words import STOP_WORDS
stopwords=list(STOP_WORDS)
import spacy
nlp = spacy.load('en_core_web_sm')

def text_cleaner(sentence):
    doc=nlp(sentence)
    
    tokens=[]
    for token in doc:
        if token.lemma_!="-PRON-":
            temp=token.lemma_.lower().strip()
        else:
            temp=token.lower_
        tokens.append(temp)
        
    cleaned_tokens=[]
    for token in tokens:
        if token not in stopwords and token not in punc:
            cleaned_tokens.append(token)
    return cleaned_tokens

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
tfidf=TfidfVectorizer(tokenizer=text_cleaner)
classifier=LinearSVC()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
clf=Pipeline([('tfidf',tfidf),('clf',classifier)])
clf.fit(x_train,y_train)

pickle.dump(clf,open("model.pkl","wb"))