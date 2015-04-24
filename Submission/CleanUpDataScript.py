#!/usr/bin/env python
# encoding: utf-8
"""
CleanUpDataScript.py

Created by Claudia Friedsam on 2014-08-06.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np
import LoadandInvestigate as li

## Load data
data_file = li.load_data('../train.csv')
# data_file = li.load_data('../test.csv')


##modify Sex Column
data_file['Gender'] = data_file['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
# print data_file.head(10)


##modify Age Column
#create median matrix
median_ages = np.zeros((2,3))
#populate the array
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = data_file[(data_file['Gender'] == i) & \
                              (data_file['Pclass'] == j+1)]['Age'].dropna().median()
#copy Age Column
data_file['AgeFill'] = data_file['Age']
#fill in missing age values
for i in range(0, 2):
    for j in range(0, 3):
        data_file.ix[ (data_file.Age.isnull()) & (data_file.Gender == i) & (data_file.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]
#take notes if age value is filled in
data_file['AgeIsNull'] = pd.isnull(data_file.Age).astype(int)
#check results
#print data_file[ data_file['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)
#print data_file.describe()


##modify Cabin Column
# copy Cabin
data_file['Cabin_t'] = data_file['Cabin']
# cells with values
data_file.loc[data_file.Cabin.notnull(),'Cabin_t'] = 1
# empty cells
data_file.loc[data_file.Cabin.isnull(),'Cabin_t'] = 0



##modify Embarked Column
# All missing Embarked -> just make them embark from most common place
if len(data_file.Embarked[ data_file.Embarked.isnull() ]) > 0:
    data_file.Embarked[ data_file.Embarked.isnull() ] = data_file.Embarked.dropna().mode().values
# map non-nan values:
data_file['Embarked_t'] = data_file.Embarked.dropna().map({'S':1,'C':2,'Q':3}).astype(int)
#set nan values to 0:
# data_file.loc[data_file.Embarked_t.isnull(),'Embarked_t'] = 0


# All the missing Fares -> assume median of their respective class
if len(data_file.Fare[ data_file.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = data_file[ data_file.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        data_file.loc[ (data_file.Fare.isnull()) & (data_file.Pclass == f+1 ), 'Fare'] = median_fare[f]



## inspect results
# print data_file.head(10)
# print data_file.describe()

## save dataframe as csv file
# data_file.to_csv('../train_mod.csv')

##drop non-numeric columns
data_file = data_file.drop(['Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis=1)

#print df.dtypes

##save numeric modified dataset
data_file.to_csv('../train_mod_num.csv')
# data_file.to_csv('../test_mod_num.csv')







