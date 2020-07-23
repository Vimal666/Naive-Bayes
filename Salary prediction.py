# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 19:35:28 2020

@author: Vimal PM
"""

#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix
from sklearn import preprocessing
#loading the dataset using pd.read_csv()
SalaryData_Test=pd.read_csv("D:\\DATA SCIENCE\\ASSIGNMENT\\Naive bayes\\SalaryData_Test.csv")
SalaryData_Train=pd.read_csv("D://DATA SCIENCE//ASSIGNMENT//Naive bayes//SalaryData_Train.csv")
#here I can see some categorical variables from the dataset's
#so next I would like to convert them into numerical format using label encoder()
#First am going to convert my train dataset
SalaryData_Train.columns
#Index(['age', 'workclass', 'education', 'educationno', 'maritalstatus',
     #  'occupation', 'relationship', 'race', 'sex', 'capitalgain',
     #  'capitalloss', 'hoursperweek', 'native', 'Salary'],
#from the train datasets variables called "workclass,education,maritalstatus,occupation,relationship,race,sex,native" are in the categorical format
#converting them using Labelencoder()
#loading label encoder
Le =preprocessing.LabelEncoder()
SalaryData_Train["Workclass"]=Le.fit_transform(SalaryData_Train["workclass"])
#removing the categorical one from dataset
SalaryData_Train=SalaryData_Train.drop("workclass",axis=1)
SalaryData_Train["Education"]=Le.fit_transform(SalaryData_Train["education"])
SalaryData_Train=SalaryData_Train.drop("education",axis=1)
SalaryData_Train["Maritalstatus"]=Le.fit_transform(SalaryData_Train["maritalstatus"])
SalaryData_Train=SalaryData_Train.drop("maritalstatus",axis=1)
SalaryData_Train["Occupation"]=Le.fit_transform(SalaryData_Train["occupation"])
SalaryData_Train=SalaryData_Train.drop("occupation",axis=1)
SalaryData_Train["Relationship"]=Le.fit_transform(SalaryData_Train["relationship"])
SalaryData_Train=SalaryData_Train.drop("relationship",axis=1)
SalaryData_Train["Race"]=Le.fit_transform(SalaryData_Train["race"])
SalaryData_Train=SalaryData_Train.drop("race",axis=1)
SalaryData_Train["Sex"]=Le.fit_transform(SalaryData_Train["sex"])
SalaryData_Train=SalaryData_Train.drop("sex",axis=1)
SalaryData_Train["Native"]=Le.fit_transform(SalaryData_Train["native"])
SalaryData_Train=SalaryData_Train.drop("native",axis=1)

#Next I would like to do the same thing for my test dataset
SalaryData_Test["Workclass"]=Le.fit_transform(SalaryData_Test["workclass"])
#removing the categorical one from dataset
SalaryData_Test=SalaryData_Test.drop("workclass",axis=1)
SalaryData_Test["Education"]=Le.fit_transform(SalaryData_Test["education"])
SalaryData_Test=SalaryData_Test.drop("education",axis=1)
SalaryData_Test["Maritalstatus"]=Le.fit_transform(SalaryData_Test["maritalstatus"])
SalaryData_Test=SalaryData_Test.drop("maritalstatus",axis=1)
SalaryData_Test["Occupation"]=Le.fit_transform(SalaryData_Test["occupation"])
SalaryData_Test=SalaryData_Test.drop("occupation",axis=1)
SalaryData_Test["Relationship"]=Le.fit_transform(SalaryData_Test["relationship"])
SalaryData_Test=SalaryData_Test.drop("relationship",axis=1)
SalaryData_Test["Race"]=Le.fit_transform(SalaryData_Test["race"])
SalaryData_Test=SalaryData_Test.drop("race",axis=1)
SalaryData_Test["Sex"]=Le.fit_transform(SalaryData_Test["sex"])
SalaryData_Test=SalaryData_Test.drop("sex",axis=1)
SalaryData_Test["Native"]=Le.fit_transform(SalaryData_Test["native"])
SalaryData_Test=SalaryData_Test.drop("native",axis=1)

#Next I'm going to sort my input varaibles and output variabel into another dataset
#for train and test
xtrain=SalaryData_Train.iloc[:,[0,1,2,3,4,6,7,8,9,10,11,12,13]]
ytrain=SalaryData_Train.iloc[:,5]
xtest=SalaryData_Test.iloc[:,[0,1,2,3,4,6,7,8,9,10,11,12,13]]
ytest=SalaryData_Test.iloc[:,5]
#model gaussianNB
ignb=GaussianNB()
#I would like to fit this model to my xtrain and ytrain to predict the xtest
pred_gnb=ignb.fit(xtrain,ytrain).predict(xtest)
#confusion matrix
confusion_matrix(ytest,pred_gnb)
#array([[10759,   601],
#       [ 2491,  1209]],
#finding the accuracy
np.mean(pred_gnb==ytest.values.flatten())
# 0.7946879150066402  (79% accuracy)

#next am going multi gaussianNB
dmnb=MultinomialNB()
#I would like to fit this model to my xtrain and ytrain to predict the xtest
pred_mnb=dmnb.fit(xtrain,ytrain).predict(xtest)
#confusionmatrix
confusion_matrix(ytest,pred_mnb)
#array([[10891,   469],
    #   [ 2920,   780]],
#finding the accuracy    
np.mean(pred_mnb==ytest.values.flatten())
# 0.7749667994687915  (77% accuracy)