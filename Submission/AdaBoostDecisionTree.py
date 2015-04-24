#!/usr/bin/env python
# encoding: utf-8
"""
AdaBoostDecisionTree.py

Created by Claudia Friedsam on 2014-08-19.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import LoadandInvestigate as li
import pandas as pd
import numpy as np
import csv as csv

# Import the random forest package
# importing necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostCLassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load modified data
train_df = li.load_data('../train_mod_num_ext2.csv')
test_df  = li.load_data('../test_mod_num_ext2.csv')
test_orig_df = li.load_data('../test.csv')
train_df.drop(['Unnamed: 0','AgeIsNull'],1,inplace=True)
test_df.drop(['Unnamed: 0','AgeIsNull'],1,inplace=True)


# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values
# Collect the test data's PassengerIds 
ids = test_orig_df['PassengerId'].values

# Convert back to a numpy array
y = train_df.Survived.values
train_df.drop('Survived',1,inplace=True)
X = train_df.values


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Fit the training data to the Survived labels and create the decision trees
print 'Training...'
# Fit regression model
clf_1 = DecisionTreeClassifier(max_depth=4)

clf_2 = AdaBoostClassifier(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300)

clf_1.fit(X_train, y_train)
clf_2.fit(X_train, y_train)

# Predict
print 'Predicting...'
y_1 = clf_1.predict(X_test).astype(int)
y_2 = clf_2.predict(X_test).astype(int)

print y_1
print y_2

#print accuracy score
acc_score = accuracy_score(y_test,y_1)
print('Accuracy Score y_1')
print(acc_score)

#print accuracy score
acc_score = accuracy_score(y_test,y_2)
print('Accuracy Score y_2')
print(acc_score)


# # Compute confusion matrix y_1
# cm = confusion_matrix(y_test, y_1)
# print('Confusion matrix y_1, without normalization')
# print(cm)
# 
# # Normalize the confusion matrix by row (i.e by the number of samples
# # in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)
# 
# print('Normalized confusion matrix')
# print(cm_normalized)
# 
# # Compute confusion matrix y_2
# cm = confusion_matrix(y_test, y_2)
# print('Confusion matrix y_2, without normalization')
# print(cm)
# 
# # Normalize the confusion matrix by row (i.e by the number of samples
# # in each class)
# cm_normalized = cm.astype('float') / cm.sum(axis=1)
# 
# print('Normalized confusion matrix')
# print(cm_normalized)

# # Take the same decision trees and run it on the test data
# print 'Predicting...'
# output = forest.predict(test_data)
# output = output.astype(int)
# 
# predictions_file = open("myownfirstforest.csv", "wb")
# open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["PassengerId","Survived"])
# open_file_object.writerows(zip(ids, output))
# predictions_file.close()
# print 'Done.'




