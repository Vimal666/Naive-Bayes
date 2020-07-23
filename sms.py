# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:03:13 2020

@author: Vimal PM
"""

#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#loading the dataset
sms = pd.read_csv("D:\\DATA SCIENCE\\ASSIGNMENT\\Naive bayes\\sms_raw_NB.csv",encoding = "ISO-8859-1")
##loading stopwords
with open("D:\DATA SCIENCE\Data sets\\stopwords.txt","r") as f:
    stopwords=f.read().split("\n")
    
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))    

sms.text = sms.text.apply(cleaning_text)

#removing empty rows
sms=sms.loc[sms.text !=" ",:]

#matrix for all text documents
def split_into_words(i):
    return [word for word in i.split(" ")]

#spliting data's into train and test data
sms_train,sms_test=train_test_split(sms,test_size=0.3)

#all email texts to count matrix
sms_bow=CountVectorizer(split_into_words).fit(sms.text)
#for all texts
all_sms_matrix=sms_bow.transform(sms.text)
all_sms_matrix.shape
#(5559, 6660)

#for training messages
train_sms_matrix=sms_bow.transform(sms_train.text)
train_sms_matrix.shape
#(3891, 6660)
#for test messages
test_sms_matrix=sms_bow.transform(sms_test.text)
test_sms_matrix.shape
# (1668, 6660)
#preparing the model and  getting the accuracy
#first am going for gaussian model
gnb=GaussianNB()
gnb.fit(train_sms_matrix.toarray(),sms_train.type.values)
#predcing the train_sms_matrix
pred_gnb=gnb.predict(train_sms_matrix.toarray())
#getting the accuracy
accuracy=np.mean(pred_gnb==sms_train.type)
#0.9103058339758416 (91%)
#next doing the same process for test sms
pred_gnb2 = gnb.predict(test_sms_matrix.toarray())
#Accuracy
accuracy2=np.mean(pred_gnb2==sms_test.type)
#0.8597122302158273 (85%)
#multinomial naive bayes model
mnb=MultinomialNB()
mnb.fit(train_sms_matrix.toarray(),sms_train.type.values)
#predicting train_sms_matrix using multinominal naive bayes
pred_mnb=mnb.predict(train_sms_matrix.toarray())
#accuracy
Accuracy=np.mean(pred_mnb==sms_train.type)
#0.9897198663582627 (98%)
#for test data
pred_mnb2=mnb.predict(test_sms_matrix.toarray())
Accuracy1=np.mean(pred_mnb2==sms_test.type)
#0.9688249400479616 (96%)
