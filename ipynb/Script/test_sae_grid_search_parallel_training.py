import sys

import numpy as np
import time
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

np.set_printoptions(3)

sys.path.append('../')

from Sonar import StackedAutoEncoderCV as SAE

if __name__ == '__main__':
    ########################################### LOAD DATA
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target

    ########################################### SELECT ONE CLASS
    knowncls = 2
    data   = data[target == knowncls]
    target = target[target == knowncls]
    nclasses = np.unique(target).shape[0]
    cls_name = dataset['target_names'][knowncls]
    ########################################### TRAINING INDEXES
    
    # Test x Development
    seed = 10
    ntrn = 0.7
    Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn, stratify=target,
                                                              random_state = seed)
    
    
    ########################################## GRID-SEARCH
    param_grid = {
        'hiddens': [[Xtrn.shape[1], 10, 5, 2, 5, 10, Xtrn.shape[1]] ],
        'optimizers': [['adam','adam','adam']],
        'nepochs': [100],
        'batch_size': [100],
        'ninit': [10]
    }
    cvmodel = SAE.StackedAutoEncoderCV(param_grid, nfolds=5, njobs = 4, random_seed = seed)
    cvmodel.fit(Xtrn, Ytrn, nclasses)
    cvmodel.save('../Models/iris_sae_' + cls_name + '.jbl')

        
    



    
    
# END OF FILE





