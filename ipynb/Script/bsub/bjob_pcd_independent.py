

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.PCD_Independent as PyPCD
import PyNN.HiddenNeuronTrain as PyHNT

from sklearn.externals import joblib
import time

if len(sys.argv) != 3:
    print 'Missing arguments to script:'
    print '> ', sys.argv[0], ' <novelty class> <cv index>'
    sys.exit(-1)
##########################################################################################
## CONFIGURATION
cvidx = int(sys.argv[2])
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
nov_classes = [sys.argv[1]]
full_classes = ['ClasseA','ClasseB','ClasseC', 'ClasseD']
classes = np.sort(np.setdiff1d(full_classes, nov_classes))
novcls = ''.join(nov_classes) # label
nPts = 400
nEvts = -1
results = {}
##########################################################################################
## TRAIN INDEXES FILE
savedir = os.getenv('SONARHOME') + '/data/'
fcrossv = savedir + 'cvindexes_'+ ''.join(nov_classes) + '_1024nfft.jbl'
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_Independent_Full/'
fsave = savedir + 'pcdind_cv_'+ ''.join(nov_classes) + '_fromMat_1024nfft.jbl'
##########################################################################################
## DATA LOADING
fname = os.getenv('SONARHOME') + '/data/' + fname
# Load and filter data
rawdata = joblib.load(fname)
for shipData in rawdata.values():
    for runData in shipData.values():
        runData['Signal'] = runData['Signal'][:nPts,:] if nEvts == -1 else runData['Signal'][:nPts,:nEvts]
        runData['Freqs'] = runData['Freqs'][:nPts]
# Pop classes not used
for ship in rawdata.keys():
    if ship in nov_classes:
        rawdata.pop(ship)
results['Novelties'] = nov_classes
##########################################################################################
## CREATE TARGET AND DATA AS MATRICES
target = None
nClass = len(rawdata.values())
results['Classes'] = classes
for iship, ship in enumerate(rawdata.keys()):
    tot = np.sum([runData['Signal'].shape[1] for runData in rawdata[ship].values()])
    if nClass == 2:
        aux = -np.ones(1)
        aux[0] = 1 if iship > 0 else -1
    else:
        aux = -np.ones(nClass)
        aux[iship] = 1
    target = np.tile(aux,(tot,1)) if target is None else np.concatenate((target, np.tile(aux,(tot,1))))
X = np.concatenate([y['Signal']for x in rawdata.values() for y in x.values()], axis=1)
data = X.transpose()
##########################################################################################
## DATA INDEXING
indexes = joblib.load(fcrossv)
##########################################################################################
## Network parameters
netPar = {'builder': nn.RProp, 'activ': 'tanh:tanh'}
##########################################################################################
## PCD Train parameters
trnPar = {
    'itrn': indexes['Indexes'][cvidx]['ITrn'],
    'ival': indexes['Indexes'][cvidx]['IVal'],
    'itst': indexes['Indexes'][cvidx]['ITst'],
    'nprocesses': 5,
    'ninit': 1,
    'fbatch': True,
    'nepochs': 100,
    'min_epochs': 50,
    'nshow': 20,
    'winit': 'inituni_ortho',
    'perftype': 'PD', # Use mean detection instead of SP
    'task': 'classification',
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
## PCD Extraction
pcdPar = {
    'NNetParameters':netPar,
    'MaxPCD' : 2,
}
pypcd = PyPCD.PCD_Independent(pcdPar)
pypcd.train(data, target, trnPar)
##########################################################################################
## Hidden Neuron Train parameters
hiddenPar = {
    'HNeurons': np.array([0]),
    'NNetParameters': netPar
}
trnPar['winit'] = 'initnw'
##########################################################################################
## LOOP OVER PCDs
clf = {}
print range(len(pypcd.PCDNum))
for npcd in range(len(pypcd.PCDNum)):
    W = pypcd.PCDNets[npcd].W[0]
    ##########################################################################################
    ## DATA PROJECTION ONTO PCDs
    data = W.dot(X).transpose()
    ##########################################################################################
    ## TRAIN CLASSIFIERS
    hiddenPar['HNeurons'][0] = npcd+1
    pyhidd = PyHNT.HiddenNeuronTrain(hiddenPar)
    pyhidd.train(data, target, trnPar)
    clf[pypcd.PCDNum[npcd]] = pyhidd
##########################################################################################
## SAVE
pcdPar.pop('NNetParameters') # must pop cause SWIG object is not picklable
hiddenPar.pop('NNetParameters')
results['Classifier'] = clf
results['PCDModel'] = pypcd
results['pcdPar'] = pcdPar
results['trnPar'] = trnPar

#joblib.dump(results, fsave, compress=9)


# end of file

