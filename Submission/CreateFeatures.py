#!/usr/bin/env python
# encoding: utf-8
"""
CreateFeatures.py

Created by Claudia Friedsam on 2014-08-13.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import pandas as pd
import numpy as np
import LoadandInvestigate as li


## load data into a dataframe and display general information
# df = li.load_data('../train_mod_num.csv')
df = li.load_data('../test_mod_num.csv')

## from the Tutorial

df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass

#create categories for Fare
# df_sorted = df.sort_index(by=['Gender','Fare'], ascending =[1,1])
# df_sorted.to_csv('../train_mod_num_sorted.csv')
# 7.5,7.8,8,9.5, 13,16,25,29,42, 77
# 1-23, 27-38, 69
# conclusion: make classes 3: <10, 2: 10-70, 1:>70
df['Fare_Cat'] = df['Fare']
df.loc[df.Fare_Cat<=10, 'Fare_Cat'] = int(3)
df.loc[(df.Fare_Cat>10) & (df.Fare_Cat<=70), 'Fare_Cat'] = int(2)
df.loc[(df.Fare_Cat>70), 'Fare_Cat'] = int(1)

#create categories for Age
# df_sorted = df.sort_index(by=['Gender','AgeFill'], ascending =[1,1])
# df_sorted.to_csv('../train_mod_num_sorted.csv')
# 16,20,22,25,26,28,31,35,41,48
# 49,30,33,30,15,17,29,30,53,18,37
#conclusion: make classes: <16: 1, 16-35: 2, >35:3
df['Age_Cat'] = df['AgeFill']
df.loc[df.Age_Cat<=15, 'Age_Cat'] = int(1)
df.loc[(df.Age_Cat>15) & (df.Age_Cat<=35), 'Age_Cat'] = int(2)
df.loc[(df.Age_Cat>35), 'Age_Cat'] = int(3)

df.drop('Unnamed: 0',1,inplace=True)
df.drop('PassengerId',1, inplace=True)

# df.to_csv('../train_mod_num_ext2.csv')
df.to_csv('../test_mod_num_ext2.csv')

df.drop(['AgeIsNull','SibSp','Parch'],1, inplace=True)
# df.to_csv('../train_mod_num_ext3.csv')
df.to_csv('../test_mod_num_ext3.csv')


