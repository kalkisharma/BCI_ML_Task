"""
@package correlation_extras.py

A few correlation functions that are not in the sklearn implementation.
The implementation style follows more or less that in sklearn.gaussian_process.correlation_models

@author Daning Huang
@date   09/19/2018
"""

import numpy as np

def matern12(theta, d):
    """Matern12 function."""
    theta = np.asarray(theta, dtype=np.float64)
    d = np.abs(np.asarray(d, dtype=np.float64))

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        td = d * theta
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        td = d * theta.reshape(1, n_features)
    rl = np.sqrt(np.sum(td*td, axis=1))

    return np.exp(-rl)

def matern32(theta, d):
    """Matern32 function."""
    theta = np.asarray(theta, dtype=np.float64)
    d = np.abs(np.asarray(d, dtype=np.float64))

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        td = d * theta
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        td = d * theta.reshape(1, n_features)
    rl = np.sqrt(3.0) * np.sqrt(np.sum(td*td, axis=1))

    return (1.0 + rl) * np.exp(-rl)

def matern52(theta, d):
    """Matern52 function."""
    theta = np.asarray(theta, dtype=np.float64)
    d = np.abs(np.asarray(d, dtype=np.float64))

    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1

    if theta.size == 1:
        td = d * theta
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        td = d * theta.reshape(1, n_features)
    rl = np.sqrt(5.0) * np.sqrt(np.sum(td*td, axis=1))

    return (1.0 + rl + rl*rl/3.0) * np.exp(-rl)
