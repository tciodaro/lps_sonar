


import neuralnet as nn

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.PCD_Constructive as PyPCD
import PyNN.CrossValidation as PyCV


# Configuration
funcs = 'tanh:tanh'

# Get Data
iris = load_iris()
data = iris['data']
target = -np.ones((data.shape[0], len(np.unique(iris['target']))))
indexes = {}
for idx, cls in enumerate(iris['target']):
    target[idx, cls] = 1.0
for cls in np.unique(iris['target']):
    indexes[cls] = np.nonzero(iris['target'] == cls)[0]
##########################################################################################
## Network parameters
netPar = {'builder': nn.RProp, 'activ': funcs}
##########################################################################################
## Train parameters
trnPar = {
    'ninit': 10,
    'fbatch': True,
    'nepochs': 100,
    'nshow': 0,
    'winit': PyInit.inituni,
    'task': 'classification',
    'datanorm': PyNorm.mapstd
}
##########################################################################################
## PCD PARAMETERS
pcdPar = {
    'NNetParameters':netPar,
    'MaxPCD' : 4,
    'EvalDiff' : -1e10,
}
##########################################################################################
## Cross Validation
cvPar = {
    'indexes': indexes,
    'TrnPerc': 0.7,
    'ValPerc': 0.3,
    'CVNSel' : 2,
    'CVNFold': 10
}
pycv = PyCV.CVFold(cvPar)
pycv.train(data, target, PyPCD.PCD_Constructive, pcdPar, trnPar)

