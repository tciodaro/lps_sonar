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
    knowncls = 1
    data   = data[target == knowncls]
    target = target[target == knowncls]
    nclasses = np.unique(target).shape[0]

    ########################################### TRAINING INDEXES
    
    # Test x Development
    ntrn = 0.7
    Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn, stratify=target)
    
    
    ########################################## GRID-SEARCH
    param_grid = {
        'hiddens': [[Xtrn.shape[1], 10, 5, 2, 5, 10, Xtrn.shape[1]],
                    [Xtrn.shape[1], 10, 5, 1, 5, 10, Xtrn.shape[1]]],
        'optimizers': [['adam','adam','adam']],
        'nepochs': [100],
        'batch_size': [100],
        'ninit': [1]
    }
    cvmodel = SAE.StackedAutoEncoderCV(param_grid, 4, 2)
    cvmodel.fit(Xtrn, Ytrn, nclasses)
    cvmodel.save('teste.jbl')

    print cvmodel.network.trn_info[1]['loss'][-1]

    loaded_model = SAE.StackedAutoEncoderCV()
    loaded_model.load('teste.jbl')
    print loaded_model.network.trn_info[1]['loss'][-1]


    # Compare
    print cvmodel.mean_score
    print loaded_model.mean_score



    raise Exception('STOP')
    
    



    
    
# END OF FILE





