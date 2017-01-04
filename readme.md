# CS 390MB Fall 2016 Group 2 A2 Part 4

### A project for CS390MB - Mobile Health Sensing and Monitoring

### [Assignment Description](http://none.cs.umass.edu/~dganesan/courses/fall16/schedule.html)

### Starter Code ([features.py](http://goo.gl/kPU3Hi), [util.py](http://goo.gl/ktC0sr),  [activity-recognition-train.py](http://goo.gl/aGwUim), [collect-labelled-activity-data.py](http://goo.gl/hxf1mf),  [activity-recognition.py](http://goo.gl/eRxlaT))

### Instructor: [Deepak Ganesan](https://people.cs.umass.edu/~dganesan/)

## Description

This trains a decision tree classifier to identify activities based on accelerometer data. This is meant to be used via an app and a server. The server is now down so the server data gathering and prediction functionality won't work. However, it is still possible to train and evaluate the classifier with some data that has already been gathered. 

## Student Implementations

All student implementations are surrounded with comments.

### `activity-classification-train.py`

Training within k-fold validation loop as well as accuracy, precision, and recall calculations.

### `activity-recognition.py`

Implementation of `predict` function.

### `features.py`

All feature extaction except `_compute_mean_features`


## Usage

### To get labeled data from the server saved to a csv

This will not work since the server is not currently running. If the server were running data could be collected and appened to `data/my-activity-data.csv` like so:

```
python collect-labelled-activity-data.py
```

This stores data in a file called **my-activity-data.csv** in the current directory. This is done in a section we can't change.

### To train the classifier

To train on data from `data/my-activity-data.csv`:

```
python activity-classification-train.py
```

This will train the classifier and serialize it into a pickle file called **classifier.pickle**. It will overwrite the file since it is there already.

### To do server-side prediction

This will not work since the server is not currently running. If the server were running, make sure you have a classifier in `classifier.pickle` and run:

```
python activity-recognition.py
```

## Files

### `data/my-activity-data.csv`

Labeled data.

### `activity-classification-train.py`

Trains a decision tree classifier on data from `data/my-activity-data.csv` and serializes the classifier to `classifier.pickle`.

### `activity-recognition.py`

Handles server-side activity recognition.

### `classifier.pickle`

A serialization of the classifier.

### `collect-labelled-activity-data.py`

Server side data collection.

### `features.py`

Extracts features.


### `util.py`

Instructor provided this to handle windows.
