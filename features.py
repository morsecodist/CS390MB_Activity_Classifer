# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.

As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).

"""

import numpy as np

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window, axis=0)

# %%---------------------------------------------------------------------------
#
#		                       Our Features
#
# -----------------------------------------------------------------------------

#### Student implemented

# Statistical Features

def _compute_variance_features(window):
    """
    Computes the variance x, y and z acceleration over the given window.
    """
    return np.var(window, axis=0)

def _compute_zero_crossing_rate_features(window):
    """
    Computes the zero crossing rate x, y and z acceleration over the given window.
    """
    return np.divide(np.sum(np.abs(np.sign(np.diff(np.sign(np.subtract(window, _compute_mean_features(window))), axis=0))), axis=0), len(window))

# Magnitude Features

def _compute_magnitude(window):
    """
    Computes the magnitude over the given window.
    """
    return np.power(np.sum(np.power(window, 2), axis=1), 0.5)

def _compute_magnitude_mean_features(window):
    """
    Computes the mean of the magnitude over the given window.
    """
    return [np.mean(_compute_magnitude(window))]

def _compute_magnitude_variance_features(window):
    """
    Computes the variance over the given window.
    """
    return [np.var(_compute_magnitude(window))]

# Entropy Features

def _compute_entropy_features(window):
    hists = [np.histogram(window[:, i], density=True)[0] for i in range(0, np.size(window, 1))]
    return [-np.nansum((i*np.log(np.abs(i)))) for i in hists]

def _compute_magnitude_entropy_features(window):
    hist = np.histogram(_compute_magnitude(window), density=True)[0]
    return [-np.nansum((hist*np.log(np.abs(hist))))]

####

def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature matrix X.

    Make sure that X is an N x d matrix, where N is the number
    of data points and d is the number of features.

    """

    x = []

    x = np.append(x, _compute_mean_features(window))

    #### Student Implemented
    # Statistical Features
    x = np.append(x, _compute_variance_features(window))
    x = np.append(x, _compute_zero_crossing_rate_features(window))

    # Magnitude features
    x = np.append(x, _compute_magnitude_mean_features(window))
    x = np.append(x, _compute_magnitude_variance_features(window))

    # Entropy features
    x = np.append(x, _compute_entropy_features(window))
    x = np.append(x, _compute_magnitude_entropy_features(window))
    ####

    return x
