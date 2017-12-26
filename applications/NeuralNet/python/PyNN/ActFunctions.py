import numpy as np

###########################################################
"""
    Possible activation functions
"""
def tanh(v):
    return np.tanh(v)
def purelin(v):
    return v
def sigmoid(v):
    return 1.0 / (1.0 + np.exp**(-1.0 * v))
def winner(v):
    return float(v >= 0)
