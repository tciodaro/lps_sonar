

import numpy as np
import os
import sys
from sklearn.externals import joblib
import PyNN.CrossValidation as PyCV


if len(sys.argv) < 4:
    print 'Missing arguments to script:'
    print '> ', sys.argv[0], ' <data file> <N> <save file> <novelty classes>'
    sys.exit(-1)
##########################################################################################
## CONFIGURATION
# Data file
fname = sys.argv[1] # 'lofar_data_1024fft_3dec_fromMat.jbl'
# Number of points to use from the spectrum
nPts = int(sys.argv[2]) # 400
# File to save the data
fsave = sys.argv[3]
# Novelty classes
nov_classes = [arg for arg in sys.argv[4:]]
##########################################################################################
## DATA LOADING
fname = os.getenv('SONARHOME') + '/data/' + fname
# Load and filter data
rawdata = joblib.load(fname)
for shipData in rawdata.values():
    for runData in shipData.values():
        runData['Signal'] = runData['Signal'][:nPts,:]
        runData['Freqs'] = runData['Freqs'][:nPts]
# Pop classes not used
nov_rawdata = {}
classes = []
for ship in rawdata.keys():
    if ship in nov_classes:
        nov_rawdata[ship] = rawdata.pop(ship)
    else:
        classes.append(ship)
data_nov = np.concatenate([y['Signal']for x in nov_rawdata.values() for y in x.values()], axis=1).transpose()
##########################################################################################
## CREATE TARGET AND DATA AS MATRICES
target = None
nClass = len(rawdata.values())
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
obj = {'data':data,
       'target': target,
       'data_nov': data_nov,
       'fdata': fname,
       'novelties': nov_classes,
       'classes': classes
}
if False:
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
    ## Cross Validation
    cvPar = {
        'indexes': indexes,
        'TrnPerc': 0.7,
        'ValPerc': 0.3,
        'CVNSel' : 10,
        'CVNFold': 10
    }
    pycv = PyCV.CVMultiFold(cvPar)
    cvidx =  pycv.get_indexes()
    cvPar['Indexes'] = cvidx
    obj['cvPar'] = cvPar

    
##########################################################################################
## SAVE FILE
fsave = os.getenv('SONARHOME') + '/data/' + fsave
joblib.dump(obj, fsave, compress=9)


# End of file


