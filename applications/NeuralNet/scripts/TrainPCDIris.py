


import neuralnet as nn

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.PCD_Constructive as PyPCDCon
import PyNN.PCD_Independent as PyPCDInd

np.set_printoptions(precision=0, suppress=True)

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
netBuilder = {'builder': nn.RProp}
##########################################################################################
## Train parameters
trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'ninit': 50,
    'fbatch': True,
    'nepochs': 200,
    'nshow': 0,
    'winit': 'inituni_ortho', #
    'task': 'classification',
    'perftype': 'PD', # Use mean detection instead of SP
    'datanorm': None #PyNorm.mapstd
}
##########################################################################################
## PCD PARAMETERS
pcdPar = {
    'NNetParameters':netBuilder,
    'MaxPCD' : 3,
    'EvalDiff' : -1e10,
}


if True:
    ##########################################################################################
    ## PCD INDEPENDENT
    pypcd = PyPCDInd.PCD_Independent(pcdPar)
    pypcd.train(data, target, trnPar)
    ##########################################################################################
    ## PCD ANGLE
    W = pypcd.PCDNets[-1].W[0]
    Cang = np.zeros((W.shape[0],W.shape[0]))
    for i in range(Cang.shape[0]):
        for j in range(Cang.shape[0]):
            dot = W[i,:].dot(W[j,:])
            norm_i = np.linalg.norm(W[i,:],2)
            norm_j = np.linalg.norm(W[j,:],2)
            cos = dot/norm_i/norm_j
            cos = cos if cos < 0.999 else 1
            Cang[i,j] = np.degrees(np.arccos(cos))
    Cang[np.isnan(Cang)] = 0
    Cang[(Cang > 90)&(Cang <= 180)] = Cang[(Cang > 90)&(Cang < 180)] - 180
    Cang[(Cang > 180)&(Cang <= 270)]= Cang[(Cang > 180)&(Cang <= 270)] - 180
    Cang = np.abs(Cang)
    print Cang

if False:    
    ##########################################################################################
    ## PCD CONSTRUCTIVE
    trnPar['winit'] = 'inituni'
    trnPar['perftype'] = 'PD'
    pypcd = PyPCDCon.PCD_Constructive(pcdPar)
    pypcd.train(data, target, trnPar)
    ##########################################################################################
    ## PCD ANGLE
    W = pypcd.PCDNets[-1].W[0]
    Cang = np.zeros((W.shape[0],W.shape[0]))
    for i in range(Cang.shape[0]):
        for j in range(Cang.shape[0]):
            dot = W[i,:].dot(W[j,:])
            norm_i = np.linalg.norm(W[i,:],2)
            norm_j = np.linalg.norm(W[j,:],2)
            cos = dot/norm_i/norm_j
            cos = cos if cos < 0.999 else 1
            Cang[i,j] = np.degrees(np.arccos(cos))
    Cang[np.isnan(Cang)] = 0
    Cang[(Cang > 90)&(Cang <= 180)] = Cang[(Cang > 90)&(Cang < 180)] - 180
    Cang[(Cang > 180)&(Cang <= 270)]= Cang[(Cang > 180)&(Cang <= 270)] - 180
    Cang = np.abs(Cang)
    print Cang