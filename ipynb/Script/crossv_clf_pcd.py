

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
import PyNN.CrossValidation as PyCV
import PyNN.HiddenNeuronTrain as PyHNT

from sklearn.externals import joblib
import time

"""
ClasseA :  SP = 91.168, PD = 91.241, MSE = 0.485  for #cv  6
ClasseB :  SP = 93.937, PD = 93.952, MSE = 0.365  for #cv  3
ClasseC :  SP = 92.948, PD = 92.993, MSE = 0.395  for #cv  3
ClasseD :  SP = 89.781, PD = 89.832, MSE = 0.563  for #cv  7
"""

##########################################################################################
## CONFIGURATION
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
nov_classes = ['ClasseD']
full_classes = ['ClasseA','ClasseB','ClasseC', 'ClasseD']
classes = np.sort(np.setdiff1d(full_classes, nov_classes))
novcls = ''.join(nov_classes) # label
nPts = 400
nEvts = -1
npcd = 4
IPCD = {
    'ClasseA': 6,
    'ClasseB': 3,
    'ClasseC': 3,
    'ClasseD': 7
}
results = {}
##########################################################################################
## Load PCD File
pcddir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_PyNN/'
if len(nov_classes):
    filepcd = pcddir + 'pcdcon_cv_'+ novcls + '_fromMat_1024nfft.jbl'
else:
    filepcd = pcddir + 'pcdcon_cv_full_fromMat_1024nfft.jbl'
PCD = joblib.load(filepcd)
PCDNet = PCD['Model'].results[IPCD[novcls]].PCDNets[npcd-1]
PCDAvg = PCDNet.avgIn
PCDStd = PCDNet.stdIn
PCDMat = PCDNet.W[0]
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_Classifier/'
if len(nov_classes):
    fsave = savedir + 'clf_pcdcon_cv_'+ novcls + '_fromMat_1024nfft_4npcd.jbl'
else:
    fsave = savedir + 'clf_pcdcon_cv_full_fromMat_1024nfft.jbl'
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
## DATA PROJECTION ONTO PCDs
data = PCDMat.dot(data.transpose()).transpose()
##########################################################################################
## DATA INDEXING
indexes = {} # one per run
offset = 0
irun = 0
for iship, ship in enumerate(rawdata):
    for runData in rawdata[ship].values():
        indexes[irun] = np.arange(runData['Signal'].shape[1]) + offset
        offset = offset + runData['Signal'].shape[1]
        irun = irun + 1
##########################################################################################
## NNet parameters
NNetPar = {'builder': nn.RProp, 'activ': 'tanh:tanh'}
##########################################################################################
## Hidden Neuron Train parameters
hiddenPar = {
    'MaxNeuron': 10,
    'MinNeuron': 2,
    'StepNeuron': 1,
    'NNetParameters': NNetPar
}
##########################################################################################
## Train parameters
trnPar = {
    'ninit': 20,
    'fbatch': True,
    'nepochs': 200,
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
pycv.train(data, target, PyHNT.HiddenNeuronTrain, hiddenPar, trnPar)

hiddenPar.pop('NNetParameters')
results['Model'] = pycv
results['cvPar'] = cvPar
results['trnPar'] = trnPar
results['hiddenPar'] = trnPar

joblib.dump(results, fsave, compress=9)


# end of file











