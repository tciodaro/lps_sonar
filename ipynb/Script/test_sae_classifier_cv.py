import sys

import numpy as np
import time
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

np.set_printoptions(3)

sys.path.append('../')

from Sonar import StackedAutoEncoderClassifierCV as SAEC
from Sonar import StackedAutoEncoderCV as SAE


if __name__ == '__main__':
    ########################################### LOAD DATA
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target
    class_names = dataset['target_names']
    ########################################### SELECT NOVELTY CLASS
    novcls = 1

    nov_data = data[target == novcls]
    nov_target = target[target == novcls]
    nov_name = class_names[novcls]

    data   = data[target != novcls]
    target = target[target != novcls]
    class_names = np.setdiff1d(class_names, [nov_name])

    ########################################### LOAD AUTO ENCODERS
    encoders = {}
    base_filename = '../Models/iris_sae_CLASS.jbl'
    for cls in class_names:
        sae = SAE.StackedAutoEncoderCV()
        sae.load(base_filename.replace('CLASS', cls))
        encoders[cls] = sae.network.get_encoder()     

    
    ########################################### TRAINING INDEXES
    
    # Test x Development
    seed = 10
    ntrn = 0.7
    Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn, stratify=target,
                                                              random_state = seed)
    
    
    ########################################## GRID-SEARCH
    param_grid = {
        'hidden': [3,4],
        'optimizer': ['adam'],
        'nepochs': [100],
        'batch_size': [100],
        'ninit': [10]
    }
    cvmodel = SAEC.StackedAutoEncoderClassifierCV(param_grid, nfolds=5, njobs = 2, random_seed = seed)
    target0_1 = ~(target == np.unique(target)[0])
    cvmodel.fit(data, target0_1, encoders)
    cvmodel.save('../Models/iris_nnet_sae_nov_' + nov_name + '.jbl')

    ########################################## TEST-NOVELTY
    thr = 0.5
    Ynov = cvmodel.predict(nov_data)
    print 'Novelty Rate: %.2f%%'%((np.max(Ynov, axis=1) > thr).sum() / float(Ynov.shape[0]))
        
    



    
    
# END OF FILE





