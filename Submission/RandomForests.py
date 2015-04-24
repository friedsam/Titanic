#!/usr/bin/env python
# encoding: utf-8
"""
RandomForests.py

Created by Claudia Friedsam on 2014-08-15.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""

import sys
import os
import LoadandInvestigate as li
import pandas as pd
import numpy as np
import csv as csv

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load modified data
train_df = li.load_data('../train_mod_num_ext2.csv')
test_df  = li.load_data('../test_mod_num_ext2.csv')
test_orig_df = li.load_data('../test.csv')
train_df.drop(['Unnamed: 0','AgeIsNull'],1,inplace=True)
test_df.drop(['Unnamed: 0','AgeIsNull'],1,inplace=True)
# train_df.drop('Unnamed: 0',1,inplace=True)
# test_df.drop('Unnamed: 0',1,inplace=True)


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


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100, max_depth = 6)

# Fit the training data to the Survived labels and create the decision trees
print 'Training...'
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the created test data
print 'Predicting...'
output = forest.predict(X_test)
output = output.astype(int)

#print accuracy score
acc_score = accuracy_score(y_test,output)
print('Accuracy Score output')
print(acc_score)

# Compute confusion matrix
cm = confusion_matrix(y_test, output)
print('Confusion matrix, without normalization')
print(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)

print('Normalized confusion matrix')
print(cm_normalized)

# Take the same decision trees and run it on the test data
print 'Predicting...'
output = forest.predict(test_data)
output = output.astype(int)

predictions_file = open("myownfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'



