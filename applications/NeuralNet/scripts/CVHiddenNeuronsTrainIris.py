

import neuralnet as nn

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.CrossValidation as PyCV
import PyNN.HiddenNeuronTrain as PyHNT

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
## NNet parameters
NNetPars = {'builder': nn.RProp, 'activ': funcs}
##########################################################################################
## Hidden Neuron Train parameters
hiddenPars = {
    'MaxNeuron': 4,
    'MinNeuron': 2,
    'StepNeuron': 1,
    'NNetParameters': NNetPars
}
##########################################################################################
## Train parameters
trnPar = {
    'ninit': 10,
    'fbatch': True,
    'nepochs': 100,
    'nshow': 0,
    'winit': PyInit.initnw,
    'task': 'classification',
    'datanorm': PyNorm.mapstd
}
##########################################################################################
## Cross Validation
TrnPerc = 0.7
ValPerc = 0.3
CVNSel = 2
CVNFold = 10
pycv = PyCV.CVFold(indexes, CVNFold, CVNSel, TrnPerc, ValPerc)
pycv.train(data, target, PyHNT.HiddenNeuronTrain, hiddenPars, trnPar)




