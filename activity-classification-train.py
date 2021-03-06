# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:02:58 2016

@author: cs390mb

Assignment 2 : Activity Recognition

This is the starter script used to train an activity recognition
classifier on accelerometer data.

See the assignment details for instructions. Basically you will train
a decision tree classifier and vary its parameters and evalute its
performance by computing the average accuracy, precision and recall
metrics over 10-fold cross-validation. You will then train another
classifier for comparison.

Once you get to part 4 of the assignment, where you will collect your
own data, change the filename to reference the file containing the
data you collected. Then retrain the classifier and choose the best
classifier to save to disk. This will be used in your final system.

Make sure to chek the assignment details, since the instructions here are
not complete.

"""

from __future__ import division
from sklearn.linear_model import LogisticRegression

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import svm
from features import extract_features # make sure features.py is in the same directory
from util import slidingWindow, reorient, reset_vars
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import pickle

from sklearn.neural_network import MLPClassifier

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = os.path.join('data', 'my-activity-data.csv')
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)


# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

# you may want to play around with the window and step sizes
window_size = 20
step_size = 20

# sampling rate for the sample data should be about 25 Hz; take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

feature_names = ["mean X", "mean Y", "mean Z", "var X", "var Y", "var Z", "zero crossing rate X", "zero crossing rate Y", "zero crossing rate Z", "magnitude mean", "magnitude var", "X entropy", "Y entropy", "Z entropy", "magnitude entropy"]
class_names = ["Sitting", "Walking", "Running", "Jumping"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

n_features = len(feature_names)

X = np.zeros((0,n_features))
y = np.zeros(0,)

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    # omit timestamp and label from accelerometer window for feature extraction:
    window = window_with_timestamp_and_label[:,1:-1]
    # extract features over window:
    x = extract_features(window)
    # append features:
    X = np.append(X, np.reshape(x, (1,-1)), axis=0)
    # append label:
    y = np.append(y, window_with_timestamp_and_label[10, -1])

print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(y)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

n = len(y)
n_classes = len(class_names)

clf = DecisionTreeClassifier(max_depth=3, max_features=5)

cv = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)

#### Student Implemented
accuracyList = []
precisionList = []
recallList = []
####

for i, (train_indexes, test_indexes) in enumerate(cv):
    print("Fold {}".format(i))

    #### Student Implemented
    clf.fit(X[train_indexes], y[train_indexes])
    conf = confusion_matrix(clf.predict(X[test_indexes]), y[test_indexes], labels=range(0, n_classes))

    accuracy = sum(sum(np.multiply(conf, np.eye(n_classes)))) / sum(sum(conf))
    accuracyList += [accuracy]

    precision = [conf[i, i] / sum(conf[:, i]) for i in range(0, n_classes)]
    precisionList += [precision]

    recall = [conf[i, i] / sum(conf[i, :]) for i in range(0, n_classes)]
    recallList += [recall]
    ####

#### Student Implemented
print "average accuracy:"
print np.nanmean(accuracyList)

print "average precision:"
print np.nanmean(precisionList, axis=0)

print "average recall:"
print np.nanmean(recallList, axis=0)
####

with open('classifier.pickle', 'wb') as f:
    pickle.dump(clf, f)
