# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.`
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os
#os.chdir("C:/Users/user/Downloads/Python_ML_DL/PythonForDataScience")
sns.set(rc={'figure.figsize':(11.7,2.27)})
cars_data=pd.read_csv('cars_sampled.csv')
cars=cars_data.copy() #deep copy change will not be reflected in the original file
cars.info()
cars.describe()
pd.set_option('display.float_format',lambda x: '%.3f' %x)
cars.describe()
pd.set_option('display.max_columns',500)
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)
cars.drop_duplicates(keep='first', inplace=True)
cars.isnull().sum()
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars)
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price']>150000)
sum(cars['price']<100)
#powePS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)
#Working Range 10 to 500
#Working Range Data
cars=cars[(cars.yearOfRegistration<=2018)
        & (cars.yearOfRegistration>=1950)
        & (cars.price>=100)
        & (cars.price<=150000)
        & (cars.powerPS>=10)
        & (cars.powerPS<=500)]
print(cars.columns)
cars['monthOfRegistration']/=12
cars['Age']=(2018-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)

cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='counts',normalize=True)
sns.countplot(x='seller',data=cars)
#Offer Types
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='counts',normalize=True)
sns.countplot(x='abtest',data=cars)

sns.boxplot(x='abtest',y='price',data=cars)
#pd.crosstab(cars['offerType'],columns='counts',normalize=True)
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='counts',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='counts',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='counts',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
cars['kilometer'].value_counts()
pd.crosstab(cars['kilometer'],columns='counts',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='counts',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='counts',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='counts',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)

#Removing insignificant variable
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#Correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
#cars_select1.corr().loc[:'price'].abs().sort_values (ascending=False)[1:] # not working 

cars_omit=cars.dropna(axis=0)
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

# imorting necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
x1=cars_omit.drop(['price'],axis='columns', inplace=False)
y1=cars_omit['price']
prices=pd.DataFrame({"1. Before":y1, "2. After":np.log(y1)})
prices.hist()
y1=np.log(y1)
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#BASELINE MODEL

base_pred=np.mean(y_test)
print(base_pred)
base_pred=np.repeat(base_pred,len(y_test))
print(base_pred)
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error) #OK

#linear regression

lgr=LinearRegression(fit_intercept=True)
model_lin1=lgr.fit(X_train,y_train)
cars_predictions_lin1=lgr.predict(X_test)

lin_mse1=mean_squared_error(y_test,cars_predictions_lin1)
print(lin_mse1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1) #OK 0.54

#R Squared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print(r2_lin_test1, r2_lin_train1) #Ok

residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1,scatter=True, fit_reg=False)
residuals1.describe()
#Results Correct

#RANDOM FOREST WITH OMITED DATA
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, max_features='auto',
                         max_depth=100, min_samples_split=10,
                         min_samples_leaf=4, random_state=1)
model_rf1=rf.fit(X_train,y_train)
cars_predictions_rf1=rf.predict(X_test)
rf_mse1=mean_squared_error(y_test, cars_predictions_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1) #OK 0.4360

r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1 ) #ok 0.8504018147750623 0.9202494705146291

#MODEL BUILDING WITH IMPUTED DATA

cars_imputed=cars.apply(lambda x: x.fillna(x.median()) \
                        if x.dtype=='float' else \
                        x.fillna(x.value_counts().index[0]))

cars_imputed.isnull().sum()
cars_imputed=pd.get_dummies(cars_imputed,drop_first=True)
x2=cars_imputed.drop(['price'],axis='columns', inplace=False)
y2=cars_imputed['price']
prices.hist()
y2=np.log(y2)
X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)
#BASELINE MODEL
base_pred=np.mean(y_test1)
print(base_pred)
base_pred=np.repeat(base_pred,len(y_test1))
print(base_pred)
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test1,base_pred))
print(base_root_mean_square_error) #OK

#linear regression

lgr2=LinearRegression(fit_intercept=True)
#Model
model_lin2=lgr2.fit(X_train1,y_train1)
cars_predictions_lin2=lgr2.predict(X_test1)

lin_mse2=mean_squared_error(y_test1,cars_predictions_lin2)
print(lin_mse2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2) #OK 0.64

#R Squared value
r2_lin_train2=model_lin2.score(X_train1,y_train1)
r2_lin_test2=model_lin2.score(X_test1,y_test1)
print(r2_lin_test2, r2_lin_train2) #Ok

# continue like that 

# END


