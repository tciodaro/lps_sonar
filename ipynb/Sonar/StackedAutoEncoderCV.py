
import time
from sklearn import metrics
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import numpy as np

from Sonar import StackedAutoEncoder as SAE


class StackedAutoEncoderCV(object):
    
    def __init__(self, grid_params, nfolds=5,njobs = 1, random_seed = None, verbose = False):
        self.grid_params = grid_params
        self.network = None
        self.grid  = None
        self.results = {}
        self.verbose = verbose
        self.nfolds = nfolds
        self.random_seed = random_seed
        self.cv_indexes = []
        self.scaler = None
        self.njobs = njobs
        self.mean_score = 1e9
        self.std_score = 1e9
           
    
    """
        Fit the grid search 
    """
    def fit(self, data, target=None, nclasses = 1):
        t0 = time.time()
        # Test x Development
        if target is None:
            target = np.ones(data.shape[0])
        kfold = None
        if nclasses == 1:
            kfold = model_selection.KFold(n_splits=4, random_state = self.random_seed)
        else:
            kfold = model_selection.StratifiedKFold(n_splits=4, random_state = self.random_seed)


        clf = Pipeline(memory=None, steps = [('scaler', preprocessing.StandardScaler()),
                                              ('network',SAE.StackedAutoEncoder(verbose=False))])

        params = dict([('network__' + k, v) for k,v in self.grid_params.items()])
        self.grid = model_selection.GridSearchCV(clf, param_grid=params, cv=kfold,
                                                 refit=False, n_jobs=self.njobs)
        self.grid.fit(data, target)
        # Find the best CV
        icv = -1
        best_score = -1e9
        for k,v in self.grid.cv_results_.items():
            if k.find('split') != -1 and k.find('_test_') != -1:
                if best_score < v[self.grid.best_index_]:
                    best_score = v[self.grid.best_index_]
                    icv = int(k[k.find('split')+5 : k.find('_')])
        # Get original indexes
        for i, (itrn, ival) in enumerate(kfold.split(data, target)):
            self.cv_indexes.append({'itrn': itrn, 'ival': ival})

        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(data[self.cv_indexes[icv]['itrn']])
        data = self.scaler.transform(data)
        # Fix parameter names
        self.grid.best_params_ = dict([(k.replace('network__',''), v) for k,v in self.grid.best_params_.items()])
        self.network = SAE.StackedAutoEncoder(**self.grid.best_params_)
        self.network.fit(data[self.cv_indexes[icv]['itrn']], target[self.cv_indexes[icv]['itrn']])
        self.mean_score = self.grid.cv_results_['mean_test_score'][self.grid.best_index_]
        self.std_score = self.grid.cv_results_['std_test_score'][self.grid.best_index_]
        print 'Total time: ', time.time()-t0
        print 'Result: %.3f +- %.3f'%(self.mean_score,self.std_score)
        

        
    def encode(self, data):
        return self.network.get_encoder().predict(self.scaler.transform(data))
    
    def predict(self, data):
        return self.network.get_auto_encoder().predict(self.scaler.transform(data))
    
    def score(self, data, target = None):
        Y = self.predict(data)
        return -self.network(Y, data)
    
    def get_network(self):
        return self.network



# end of file


