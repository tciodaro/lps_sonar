import sys

import numpy as np
import time
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

np.set_printoptions(3)

sys.path.append('../')

from Sonar import StackedAutoEncoder as SAE

if __name__ == '__main__':
    ########################################### LOAD DATA
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target

    ########################################### SELECT NOVELTY
    novcls = 2
    nov_data = data[target==novcls]
    nov_target = target[target==novcls]
    #
    data   = data[target != novcls]
    target = target[target != novcls]

    ########################################### TRAINING INDEXES
    
    # Test x Development
    ntrn = 0.7
    Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn, stratify=target)

    cv_indexes = []
    kfold = model_selection.StratifiedKFold(n_splits=4, random_state=10)

    ########################################### PREPROCESSING
    scaler = preprocessing.StandardScaler().fit(Xtrn)
    Xtrn = scaler.transform(Xtrn)
    nov_data = scaler.transform(nov_data)

    ########################################## CONFIGURATION
    hiddens = []
    optimizers = ['adam','adam','adam']
    nepochs = 500
    batch_size = 100
    ninit = 10
    
    ########################################## CROSS-VALIDATION
    t0 = time.time()
    param_grid = {
        'hiddens': [[Xtrn.shape[1], 10, 5, 2, 5, 10, Xtrn.shape[1]],
                    [Xtrn.shape[1], 10, 5, 1, 5, 10, Xtrn.shape[1]]]
    }
    clf = SAE.StackedAutoEncoder([], optimizers, nepochs, batch_size, ninit, verbose=False)
    grid = model_selection.GridSearchCV(clf, param_grid=param_grid, cv=kfold, refit=False, n_jobs=1)
    grid.fit(Xtrn, Ytrn)
    # Find the best CV
    icv = -1
    best_score = -1e9
    for k,v in grid.cv_results_.items():
        if k.find('split') != -1 and k.find('_test_') != -1:
            if best_score < v[grid.best_index_]:
                best_score = v[grid.best_index_]
                icv = int(k[k.find('split')+5 : k.find('_')])
    # Get original indexes
    for i, (itrn, ival) in enumerate(kfold.split(Xtrn, Ytrn)):
        if i == icv:
            break
    clf = SAE.StackedAutoEncoder(grid.best_params_['hiddens'], optimizers, nepochs, batch_size, ninit, verbose=False)            
    clf.fit(Xtrn[itrn], Ytrn[itrn])
    print 'Time: ', time.time()-t0
    print clf.score(Xtrn[ival], Xtrn[ival])

# END OF FILE





