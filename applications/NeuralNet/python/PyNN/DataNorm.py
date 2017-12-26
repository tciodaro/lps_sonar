

import numpy as np
from sklearn import preprocessing


def get_func(name):
    if name is None: return no_norm
    if name.lower() == 'mapstd': return mapstd
    return None




def no_norm(X, itrn = None):
    return X
    

def mapstd(X, itrn = None):

    if itrn is None:
        itrn = np.arange(X.shape[0])

    return (X - np.mean(X[itrn], axis=0)) / np.std(X[itrn], axis=0)

