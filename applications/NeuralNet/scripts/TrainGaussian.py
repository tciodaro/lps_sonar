

import neuralnet as nn
import numpy as np
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import time


# Configuration
nodes = '2:2:1'
funcs = 'tanh:tanh'

# Get Data
C1 = np.random.normal(0.0, 1.0, (100, 2))
C2 = np.random.normal(1.0, 0.5, (100, 2))
data = np.concatenate((C1,C2))
target = np.concatenate((np.ones((C1.shape[0],1)),
                        -np.ones((C1.shape[0],1))), axis=0)
# Create train indexes (train and test only)
TrnPerc = 0.7
itrn = []
itst = []
for c in [-1,1]:
    idx = np.nonzero(target[:,0] == c)[0]
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
    'ninit': 10,
    'fbatch': True,
    'nepochs': 100,
    'nshow': 20,
    'winit': PyInit.initnw,
    'task': 'classification',
    'datanorm': PyNorm.mapstd
}
t0 = time.time()
pynn.train(data, target, trnPar)
print 'Train took: %.3f s'%(time.time()-t0)

##########################################################################################
## Plot
#import matplotlib.pyplot as plt
#plt.plot(C1[:,0], C1[:,1], 'ok')
#plt.plot(C2[:,0], C2[:,1], '.r')
#plt.show()








