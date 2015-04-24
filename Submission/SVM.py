#!/usr/bin/env python
# encoding: utf-8
"""
SVM.py

Created by Claudia Friedsam on 2014-08-18.
Copyright (c) 2014 __MyCompanyName__. All rights reserved.
"""
print(__doc__)
import sys
import os
import numpy as np
import csv as csv

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import LoadandInvestigate as li

#load modified data
train_df = li.load_data('../train_mod_num_ext3.csv')
test_df  = li.load_data('../test_mod_num_ext3.csv')
test_orig_df = li.load_data('../test.csv')
train_df.drop('Unnamed: 0',1,inplace=True)
test_df.drop('Unnamed: 0',1,inplace=True)

# Convert back to a numpy array
y = train_df.Survived.values
train_df.drop('Survived',1,inplace=True)
X = train_df[['Pclass','Fare','Gender','AgeFill']].values
test_data = test_df[['Pclass','Fare','Gender','AgeFill']].values
# Collect the test data's PassengerIds 
ids = test_orig_df['PassengerId'].values

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print y_test.sum()



# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=10)
# classifier = svm.SVC(kernel='rbf', gamma=10)
# classifier = svm.NuSVC(kernel='linear',nu=0.5)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


#print accuracy score
acc_score = accuracy_score(y_test,y_pred)
print('Accuracy Score')
print(acc_score)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)

# Show confusion matrix in a separate window
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
# plt.title('Confusion matrix')
# plt.set_cmap('Blues')
# plt.colorbar()
# tick_marks = np.arange(2)
# plt.xticks(tick_marks, ['Dead','Survived'], rotation=60)
# plt.yticks(tick_marks, ['Dead','Survived'])
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# # Convenience function to adjust plot parameters for a clear layout.
# plt.tight_layout()

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)

print('Normalized confusion matrix')
print(cm_normalized)

# Show normalized confusion matrix in a separate window
# plt.figure()
# plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)
# plt.title('Normalized confusion matrix')
# plt.set_cmap('Blues')
# plt.colorbar()
# plt.xticks(tick_marks, ['Dead','Survived'], rotation=60)
# plt.yticks(tick_marks, ['Dead','Survived'])
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.tight_layout()
# plt.show()

# Take the same SVM and run it on the submission test data
print 'Predicting...'
output = classifier.fit(X_train, y_train).predict(test_data)
output = output.astype(int)

predictions_file = open("SVM.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'