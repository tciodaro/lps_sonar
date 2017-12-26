

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.CrossValidation as PyCV
import PyNN.PCD_Constructive as PyPCDCons
import PyNN.PCD_Independent as PyPCDInd

from sklearn.externals import joblib
import time

##########################################################################################
## CONFIGURATION
index_file = os.getenv('SONARHOME') + '/data/willian_thesis_indexes_1024fft.jbl'
index_file = None
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
nov_classes = ['ClasseA']
full_classes = ['ClasseA','ClasseB','ClasseC', 'ClasseD']
classes = np.sort(np.setdiff1d(full_classes, nov_classes))
novcls = ''.join(nov_classes) # label
nPts = 400
nEvts = -1
results = {}
trnPerc = 0.7
valPerc = 0.3
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/'
if len(nov_classes):
    fsave = savedir + 'pcdind_'+ ''.join(nov_classes) + '_fromMat.jbl'
else:
    fsave = savedir + 'pcdind_nonnovelty_fromMat.jbl'

##########################################################################################
## DATA LOADING
fname = os.getenv('SONARHOME') + '/data/' + fname
# Load and filter data
data = joblib.load(fname)
for shipData in data.values():
    for runData in shipData.values():
        runData['Signal'] = runData['Signal'][:nPts,:] if nEvts == -1 else runData['Signal'][:nPts,:nEvts]
        runData['Freqs'] = runData['Freqs'][:nPts]
# Pop classes not used
novs = []
for ship in data.keys():
    if ship not in classes:
        data.pop(ship)
        novs.append(ship)
results['Novelties'] = nov_classes
##########################################################################################
## PREPARE DATA FOR THE NN TRAIN AND INDEXES
X = None
itrn = []
itst = []
ival = []
offset = 0
T = None
nClass = len(data.values())
results['Classes'] = []
for iship, ship in enumerate(data.keys()):
    tot = 0
    results['Classes'].append(ship)
    for irun, runData in data[ship].iteritems():
        if index_file is None:
            idx = np.arange(0, runData['Signal'].shape[1]) + offset
            np.random.shuffle(idx)
            ntrn = int(idx.shape[0] * trnPerc) + 1
            nval = int(idx.shape[0] * valPerc) + 1
            itrn += idx[:ntrn].tolist()
            ival += idx[ntrn:ntrn+nval].tolist()
            itst += idx[ntrn+nval:].tolist()
        else: # From file
            indexes = joblib.load(index_file)
            itrn += (np.array(indexes['Train'][ship][irun]) + offset).tolist()
            ival += (np.array(indexes['Validation'][ship][irun]) + offset).tolist()
            itst += (np.array(indexes['Test'][ship][irun]) + offset).tolist()
        offset = runData['Signal'].shape[1] + offset
        tot = tot + runData['Signal'].shape[1]
    if nClass == 2:
        aux = -np.ones(1)
        aux[0] = 1 if iship > 0 else -1
    else:
        aux = -np.ones(nClass)
        aux[iship] = 1

    T = np.tile(aux,(tot,1)) if T is None else np.concatenate((T, np.tile(aux,(tot,1))))
X = np.concatenate([y['Signal']for x in data.values() for y in x.values()], axis=1)
data = X.transpose() # Transpose X, so events are rows (T is already correct)
target = T
if not len(itst): itst = ival
##########################################################################################
## Network parameters
netPar = {'builder': nn.RProp, 'activ': 'tanh:tanh'}
##########################################################################################
## Train parameters
trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'ninit': 1,
    'fbatch': True,
    'nepochs': 200,
    'min_epochs': 50,
    'nshow': 0,
    'winit': PyInit.inituni,
    'task': 'classification',
    'datanorm': PyNorm.mapstd
}
##########################################################################################
## PCD PARAMETERS
pcdPar = {
    'NNetParameters':netPar,
    'MaxPCD' : 5,
    'EvalDiff' : -1e10,
}
#pypcd = PyPCDCons.PCD_Constructive(pcdPar)
pypcd = PyPCDInd.PCD_Independent(pcdPar)
pypcd.train(data, target,trnPar)
results['Model'] = pypcd
results['pcdPar'] = pcdPar
results['trnPar'] = trnPar

pcdPar.pop('NNetParameters')
joblib.dump(results, fsave, compress=9)



# end of file







