

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.CrossValidation as PyCV
import PyNN.KDECluster as PyKDE

from sklearn.externals import joblib
import time


if len(sys.argv) != 2:
    print 'ERROR: usage'
    print sys.argv[0], ' <novelty>'
    sys.exit(-1)

###############################################################################
## CONFIGURATION
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
nov_classes = [sys.argv[1]]
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
###############################################################################
## Load PCD File
pcddir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_PyNN_WrongTrnIndex/'
if len(nov_classes):
    filepcd = pcddir + 'pcdcon_cv_'+ novcls + '_fromMat_1024nfft.jbl'
else:
    filepcd = pcddir + 'pcdcon_cv_full_fromMat_1024nfft.jbl'
PCD = joblib.load(filepcd)
PCDNet = PCD['Model'].results[IPCD[novcls]].PCDNets[npcd-1]
PCDAvg = PCDNet.avgIn
PCDStd = PCDNet.stdIn
PCDMat = PCDNet.W[0]
###############################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_KDECluster/'
if len(nov_classes):
    fsave = savedir + 'kdeclus_pcdcons_cv_'+ novcls + '_fromMat_1024nfft.jbl'
else:
    fsave = savedir + 'kdeclus_pcdcons_cv_full_fromMat_1024nfft.jbl'
###############################################################################
## DATA LOADING
fname = os.getenv('SONARHOME') + '/data/' + fname
# Load and filter data
rawdata = joblib.load(fname)
for shipData in rawdata.values():
    for runData in shipData.values():
        runData['Signal'] = runData['Signal'][:nPts,:] if nEvts == -1 else runData['Signal'][:nPts,:nEvts]
        runData['Freqs'] = runData['Freqs'][:nPts]
# Pop classes not used
nov_rawdata = {}
for ship in rawdata.keys():
    if ship in nov_classes:
        nov_rawdata[ship] = rawdata.pop(ship)
results['Novelties'] = nov_classes
data_nov = np.concatenate([y['Signal']for x in nov_rawdata.values() for y in x.values()], axis=1).transpose()
###############################################################################
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
known_targets = np.zeros(data.shape[0])
###############################################################################
## DATA PROJECTION ONTO PCDs
data = (data - PCDAvg) / PCDStd
data = PCDMat.dot(data.transpose()).transpose()
data_nov = (data_nov - PCDAvg) / PCDStd
data_nov = PCDMat.dot(data_nov.transpose()).transpose()
###############################################################################
## DATA INDEXING
indexes = {} # one per run
offset = 0
irun = 0
for iship, ship in enumerate(rawdata):
    for runData in rawdata[ship].values():
        indexes[irun] = np.arange(runData['Signal'].shape[1]) + offset
        offset = offset + runData['Signal'].shape[1]
        irun = irun + 1
###############################################################################
## Training Parameters
trnPar = {
    'kde_nsamples': 1000,
    'trn_nsamples': 1000,
    'knowledge_thr': 0.90,
    'kde_bandwidth': 1.0,
    'kde_type': 'gaussian'
}
###############################################################################
## Cross Validation
cvPar = {
    'indexes': indexes,
    'TrnPerc': 0.7,
    'ValPerc': 0.3,
    'CVNSel' : 1,
    'CVNFold': 10
}
###############################################################################
## Train Model
pykde = PyKDE.KDECluster()
pykde.train(data, known_targets, trnPar)
#pycv = PyCV.CVMultiFold(cvPar)
#pycv.train(data, known_targets, PyART.ART, {}, trnPar)

###############################################################################
## Save
#joblib.dump(gridSearch, fsave, compress=9)

# end of file











