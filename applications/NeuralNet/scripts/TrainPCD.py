


import neuralnet as nn
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_iris
import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.PCD_Constructive as PyPCDCon
import PyNN.PCD_Independent as PyPCDInd
import PyNN.PCD_IsoCooperative as PyPCDIsoCoop

np.set_printoptions(precision=0, suppress=True)

##########################################################################################
## Data
C1 = np.random.rand(1000,2) * 2 + 1

C21 = np.random.rand(500,2)
C21[:,1] = C21[:,1] * 2 + 1
C21[:,0] = C21[:,0] - 0.2
C22 = np.random.rand(500,2) * 2 + 1
C22[:,1] = C22[:,1] + 2.5
C2 = np.vstack((C21, C22))
#C2 = C22

data = np.vstack((C1, C2))

data = (data - np.mean(data,0)) / np.std(data, 0)

target = np.vstack((np.ones((C1.shape[0],1)), -np.ones((C2.shape[0],1))))
itrn = np.arange(data.shape[0])
itst = itrn
ival = itrn

#plt.plot(C1[:,0], C1[:,1], 'or')
#plt.plot(C2[:,0], C2[:,1], 'ok')
#raise Exception('STOP')

##########################################################################################
## Network parameters
netBuilder = {'builder': nn.RProp}
##########################################################################################
## Train parameters
trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'ninit': 1,
    'fbatch': True,
    'nepochs': 500,
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
    'MaxPCD' : 1,
    'EvalDiff' : -1e10,
}

##########################################################################################
## PCD INDEPENDENT
if False:
    print 'Training PCD Independent'
    pypcd = PyPCDInd.PCD_Independent(pcdPar)
    pypcd.train(data, target, trnPar)

##########################################################################################
## PCD CONSTRUCTIVE
if True:
    print 'Training PCD Constructive'
    trnPar['winit'] = 'inituni'
    trnPar['perftype'] = 'PD'
    pypcd = PyPCDCon.PCD_Constructive(pcdPar)
    pypcd.train(data, target, trnPar)

##########################################################################################
## PCD CALOBA
if True:
    print 'Training PCD IsoCooperative'
    trnPar['winit'] = 'inituni'
    trnPar['perftype'] = 'PD'
    trnPar['nhidden'] = -1
    pypcd = PyPCDIsoCoop.PCD_IsoCooperative(pcdPar)
    pynet = pypcd.train(data, target, trnPar)



raise Exception('STOP')

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
for i in range(Cang.shape[0]):
    for j in range(Cang.shape[1]):
        print '  %i'%Cang[i,j],
    print ''

# DEBUG
plt.close('all')
plt.figure()
plt.plot(data[target[:,0]==1,0], data[target[:,0]==1,1], '.k', label='C1')
plt.plot(data[target[:,0]==-1,0], data[target[:,0]==-1,1], '.r', label='C2')
plt.xlabel('D1')
plt.ylabel('D2')
ax = plt.axis()
for i in range(W.shape[0]):
    plt.plot([0, W[i,0]], [0, W[i,1]], '--', lw=3, label='W%i'%(i+1))
plt.legend(numpoints=1, loc='best')

# Projections
trndata = np.array(data)
for i in range(W.shape[0]):
    plt.figure()
    trndata = trndata - np.outer(np.array([W[i]]).T, trndata.dot(W[i])).T
    plt.plot(trndata[target[:,0]==1,0], trndata[target[:,0]==1,1], '.k')
    plt.plot(trndata[target[:,0]==-1,0], trndata[target[:,0]==-1,1], '.r')
    plt.axis(ax)
    plt.title('Data After Projection Removal From W%i'%(i+1))
    plt.xlabel('D1')
    plt.ylabel('D2')
