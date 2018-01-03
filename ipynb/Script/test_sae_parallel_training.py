import sys

import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

import multiprocessing as mp

np.set_printoptions(3)

sys.path.append('../')


def training_parallel(args):
    import Sonar.StackedAutoEncoder as SAE
    model = None
    data = args['data']
    net_config = args['net_config']
    model = SAE.StackedAutoEncoder(net_config)
    model.fit(data)
    return (model,)


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
    Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn, stratify=target, random_state=10)
    cv_indexes = []
    kfold = model_selection.StratifiedKFold(n_splits=4, random_state=10)
    for itrn, ival in kfold.split(Xtrn, Ytrn):
        cv_indexes.append({'itrn': itrn, 'ival': ival})

    ########################################### PREPROCESSING
    scaler = preprocessing.StandardScaler().fit(Xtrn)
    Xtrn = scaler.transform(Xtrn)
    nov_data = scaler.transform(nov_data)

    ########################################## CONFIGURATION
    net_config = {
        'hiddens': [Xtrn.shape[1], 10, 5, 2, 5, 10, Xtrn.shape[1]],
        'optimizers': ['adam','adam','adam'],
        'nepochs': 100,
        'batch_size': 100,
        'ninit': 1,    
        'patience' : 100,
        'verbose': 0,
    }
    
    ########################################## CROSS-VALIDATION
    results = []
    pool = mp.Pool(processes = 2)
    for icv in range(len(cv_indexes)):
        net_config['itrn'] = cv_indexes[icv]['itrn']
        net_config['ival'] = cv_indexes[icv]['ival']
        trn_args = {
            'net_config' : net_config,
            'data' : Xtrn
        }
        # POOL
        results.append(pool.apply_async(training_parallel, args=(trn_args,)))
    # GET RESULTS
    results = [p.get() for p in results]
    scores  = [obj[0].training_error for obj in results]
    print(scores)

# END OF FILE





