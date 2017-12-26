

import neuralnet as nn
#import pynnet as nn

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm

# Configuration
nodes = '4:4:3'
funcs = 'tanh:tanh'

# Get Data
iris = load_iris()
data = iris['data']
target = -np.ones((data.shape[0], len(np.unique(iris['target']))))
for idx, cls in enumerate(iris['target']):
    target[idx, cls] = 1.0

# Create train indexes (train and test only)
TrnPerc = 0.7
itrn = []
itst = []
for c in np.unique(iris['target']):
    idx = np.nonzero(iris['target'] == c)[0]
    np.random.shuffle(idx)
    n = int(idx.shape[0] * TrnPerc) + 1
    itrn += idx.tolist()[:n]
    itst += idx.tolist()[n:]
ival = itst
##########################################################################################
## Network parameters
netBuilder = {'builder': nn.RProp, 'nodes': nodes, 'activ': funcs}
pynn = PyNNet.NeuralNet(netBuilder)
##########################################################################################
## Train parameters
trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'ninit': 2,
    'nprocesses': 2,
    'fbatch': True,
    'nepochs': 200,
    'nshow': 0,
    'winit':'inituni',
    'task': 'classification',
    'datanorm': 'mapstd'
}
pynn.train(data, target, trnPar)
#pynn.write('./iris_net.jbl')
print 'Performance: ', pynn.trn_info.perf
pynn.print_weights()
