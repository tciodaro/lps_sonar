

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm

from sklearn.externals import joblib
import time

if len(sys.argv) != 4:
    print 'Missing arguments to script:'
    print '> ', sys.argv[0], ' <novelty class> <cv index> <bottle-neck>'
    sys.exit(-1)
##########################################################################################
## CONFIGURATION
cvidx = int(sys.argv[2])
nov_class = sys.argv[1]
nbottle = int(sys.argv[3])
full_classes = ['ClasseA','ClasseB','ClasseC', 'ClasseD']
classes = np.sort(np.setdiff1d(full_classes, nov_class))
nPts = 400
nEvts = -1
results = {}
sonardatadir = os.getenv('SONARDATA')
##########################################################################################
## TRAIN INDEXES FILE
fcrossv = sonardatadir + '/cvindexes_'+ nov_class + '_1024nfft.jbl'
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/Autoencoder/'
fsave = savedir + 'autoencoder_cv_'+ nov_class + '_1024nfft.jbl'
##########################################################################################
## DATA LOADING
fdata = sonardatadir + '/novelty_' + nov_class + '_1024nfft.jbl'
obj = joblib.load(fdata)
data = obj['data']
target = obj['target']
data_nov = obj['data_nov']
##########################################################################################
## DATA INDEXING
indexes = joblib.load(fcrossv)
##########################################################################################
## Network parameters
netPar = {
    'builder': nn.RProp,
    'nodes': str(data.shape[1])+':'+str(nbottle)+':'+str(data.shape[1]),
    'activ': 'tanh:lin'}
##########################################################################################
## Train parameters
trnPar = {
    'itrn': indexes['Indexes'][cvidx]['ITrn'],
    'ival': indexes['Indexes'][cvidx]['IVal'],
    'itst': indexes['Indexes'][cvidx]['ITst'],
    'nprocesses': 1,
    'ninit': 1,
    'fbatch': True,
    'nepochs': 1000,
    'min_epochs': 50,
    'nshow': 50,
    'winit': 'initnw',
    'perftype': 'MSE', 
    'task': 'estimation',
    'datanorm': 'mapstd'
}
##########################################################################################
## Replicate small class
nEvtPerClass = (target == 1).sum(axis=0)
maxClass = nEvtPerClass.max()
replicateClasses = np.floor(float(maxClass) / nEvtPerClass)
itrn = np.array(trnPar['itrn'])
for icls, totcls in enumerate(replicateClasses):
    if totcls <= 1.0: continue
    clsidx = target[itrn,icls] == 1.0
    itrn = np.concatenate((itrn, np.tile(itrn[clsidx], (totcls-1))))
trnPar['itrn'] = itrn.tolist()
##########################################################################################
## TRAINING PARAMETERS
pynet = PyNNet.NeuralNet(netPar)
pynet.train(data, data, trnPar)




