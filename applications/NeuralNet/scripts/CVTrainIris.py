

import neuralnet as nn

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.CrossValidation as PyCV

# Configuration
nodes = '4:4:3'
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
## Model
modelPar = {'builder': nn.RProp, 'nodes': nodes, 'activ': funcs}
##########################################################################################
## Train parameters
trnPar = {
    'ninit': 2,
    'fbatch': True,
    'nepochs': 100,
    'nshow': 0,
    'winit': PyInit.initnw,
    'task': 'classification',
    'datanorm': PyNorm.mapstd
}
##########################################################################################
## Cross Validation
cvPar = {
    'indexes': indexes,
    'TrnPerc': 0.7,
    'ValPerc': 0.3,
    'CVNSel' : 10,
    'CVNFold': 10
}
pycv = PyCV.CVMultiFold(cvPar)
pycv.train(data, target, PyNNet.NeuralNet, modelPar, trnPar)




