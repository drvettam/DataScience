# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:24:59 2020

@author: G M
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


data_income=pd.read_csv('income.csv')
#creating a copy of data
data=data_income.copy()

# #Exploratory Data Analysis
#To Check Variable Data Type
data.info()
data.isnull()
data.isnull().sum() #no columns has missing value ok

summary_num=data.describe()
print(summary_num)

# #............OK
summary_cate=data.describe(include="O") # Capital O - for object
print(summary_cate)

data['JobType'].value_counts()
data['occupation'].value_counts()
# #............OK
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

# #Go back and read the data by including "na_values[ ?]
data=pd.read_csv('income.csv',na_values=[" ?"])

# #Data Preprocessing
data.isnull().sum()

missing=data[data.isnull().any(axis=1)]

data2=data.dropna(axis=0)

correlation=data2.corr()
round(correlation,3)

# #Cross Table
data2.columns
gender=pd.crosstab(index=data2["gender"],columns='count', normalize=True)
print(gender)
gender_salstat=pd.crosstab(index=data2["gender"],
                            columns=data2['SalStat'],
                            margins=True,
                            normalize='index')
print(gender_salstat)

#Frequency distribution of Salary status
SalStat=sns.countplot(data2['SalStat'])

jobtype_salstat=pd.crosstab(index=data2['JobType'], 
                             columns=data2['SalStat'], 
                             margins=True,
                             normalize='index')


jobtype_salstat1=round(jobtype_salstat*100,2)

print(jobtype_salstat1)

sns.distplot(data2['age'],bins=10,kde=False)
sns.countplot(x="JobType",data=data2)

sns.boxplot('SalStat', 'age',data=data2)
data2.groupby('SalStat')['age'].median()
print(data2['SalStat'])

#Exploratory Data Analysis
sns.countplot(x=data2['JobType'],hue=data2["SalStat"]) 
sns.countplot(x=data2['EdType'],hue=data2["SalStat"]) 
sns.countplot(x=data2['occupation'],hue=data2["SalStat"]) 
# data=pd.read_csv('income.csv',na_values=[" ?"])
# data2=data.dropna(axis=0)

# ##########################
# # CaseStudy Clasification Part II
# #LOGISTIC REGRESSION
# ##########################
# #Reindexing Salary Status to 0 and 1

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1})
print(data2['SalStat'])
data2.info()
# new_data=pd.get_dummies(data2, drop_first=True) #OK
# # #Storing the column names
# columns_list=list(new_data.columns)
# print(columns_list)

# features=list(set(columns_list)-set(['SalStat'])) # Working

# print(features)

# y=new_data['SalStat'].values
# print(y)
# x=new_data[features].values
# print(x)
# #Spliting tghe data into train and test
# train_x,test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=0)

# logistic=LogisticRegression()
# logistic.fit(train_x,train_y)
# logistic.coef_

# logistic.intercept_

# prediction=logistic.predict(test_x)
# print(prediction)

# #CONFUSION MATRIX
# confusion_matrix=confusion_matrix(test_y,prediction)
# print(confusion_matrix) # result slightly different
# #Calculating the accuracy
# accuracy_score=accuracy_score(test_y,prediction)
# print(accuracy_score) # OK 0.8366 slightly different from lecture result 84.35

# #Printing misclassified values from the prediction
# print("Misclassified samples: %d" % (test_y != prediction).sum())

#LOGISTIC REGRESSION -REMOVING INSIGNIFICANT VARIABLES
#Block the codes used in the First Part of Logistic Regression before Run it from the begining
# Otherwise error "numpy.ndarray' object is not callable"

# data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0, ' greater than 50,000':1}) NOT TO RECONVERT
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)
new_data=pd.get_dummies(new_data, drop_first=True) #OK

# #Storing the column names
columns_list=list(new_data.columns)
print(columns_list)

features=list(set(columns_list)-set(['SalStat'])) # Working

print(features)

y=new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)
#Spliting tghe data into train and test
train_x,test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=0)

logistic=LogisticRegression()
logistic.fit(train_x,train_y)
logistic.coef_

logistic.intercept_

prediction=logistic.predict(test_x)

#CONFUSION MATRIX
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix) # result slightly different
#Calculating the accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score) # OK 0.8366 slightly different from lecture result 84.35

#Printing misclassified values from the prediction
print("Misclassified samples: %d" % (test_y != prediction).sum())

# #Separating tne input names from data


# print(features)
# #Storing ouput values in y
# y=new_data['SalStat'].values
# print(y)
# x=new_data[features].values
# print(x)
# train_x,test_x, train_y, test_y=train_test_split(x,y,test_size=0.3,random_state=0)
# #Make an instance of the Model
# logistic=LogisticRegression()
# logistic.fit(train_x,train_y)
# logistic.coef_
# logistic.intercept_
# prediction=logistic.predict(test_x)
# #print(prediction)


# #Result not correct


# #KNN Model
# ###############
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt

KNN_clasifier=KNeighborsClassifier (n_neighbors=5)
KNN_clasifier.fit(train_x,train_y)

prediction=KNN_clasifier.predict(test_x)
confusion_matrix1=confusion_matrix(test_y, prediction)
print("\t", "Predicted Values")
print("Original Values", "\n", confusion_matrix1)
accuracy_score1=accuracy_score(test_y,prediction)
print(accuracy_score1)

print("Misclassified samples: %d" % (test_y != prediction).sum())

Misclassified_sample=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x, train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y != pred_i).sum())

print("Misclassified samples: %d", Misclassified_sample)
