

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.TrnInfo as TrnInfo
from sklearn import preprocessing

from sklearn.externals import joblib
from sklearn import mixture
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
fcrossv = savedir + 'cvindexes_'+ novcls + '_1024nfft.jbl'
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/GMM/'
fsave = savedir + 'gmm_pcdcons_' + novcls + '_1024nfft_icv%i.jbl'%cvidx
##########################################################################################
## Load PCD File
pcddir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD/'
filepcd = pcddir + 'pcdcons_'+ novcls + '_1024nfft.jbl'
PCD = joblib.load(filepcd)
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
nov_rawdata = {}
for ship in rawdata.keys():
    if ship in nov_classes:
        nov_rawdata[ship] = rawdata.pop(ship)
results['Novelties'] = nov_classes
data_nov = np.concatenate([y['Signal']for x in nov_rawdata.values() for y in x.values()], axis=1).transpose()
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
###############################################################################
## DATA INDEXING
indexes = joblib.load(fcrossv)
###############################################################################
## Training Parameters
trnPar = {
    'itrn': indexes['Indexes'][cvidx]['ITrn'],
    'ival': indexes['Indexes'][cvidx]['IVal'],
    'itst': indexes['Indexes'][cvidx]['ITst'],
    'trn_max_it' : 100,
    'trn_min_cov': 0.01,
    'trn_max_gaussians': 10,
    'trn_max_init': 10,
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
## Train Model
results = {}
results['gmmPar'] = trnPar
results['GMM'] = {}
##########################################################################################
## LOOP OVER PCDs
#for npcd in range(len(PCD['PCDModel'].results[cvidx].PCDNets)):
for npcd in [4]:
    print '==> Training for ', npcd+1, ' PCDs'
    #############################################################################
    ## DATA PROJECTION ONTO PCDs
    PCDNet = PCD['PCDModel'].results[cvidx].PCDNets[npcd]
    PCDAvg = PCDNet.avgIn
    PCDStd = PCDNet.stdIn
    PCDMat = PCDNet.W[0]
    X = PCDMat.dot(data.transpose()).transpose()
    Xnov = PCDMat.dot(data_nov.transpose()).transpose()
    # TRAIN
    results['GMM'][npcd] = {}
    for ngauss in range(2, trnPar['trn_max_gaussians']):
        gmm = mixture.GMM(n_components=ngauss, covariance_type='full', tol=0,
                          n_iter=trnPar['trn_max_it'], n_init=trnPar['trn_max_init'],
                          params='wmc',init_params='wmc', verbose=0,
                          min_covar=trnPar['trn_min_cov'])
    gmm.fit(X[trnPar['itrn']])
    # Train SP
    Y = gmm.fit(
    trninfo = TrnInfo()
    
    gmm_coo_bic[iclus] = gmm.bic(Xpcd_coo.values[np.ix_(itrn, idx)])
    gmm_coo_aic[iclus] = gmm.aic(Xpcd_coo.values[np.ix_(itrn, idx)])
    gmm_coo[iclus] = gmm
    
    pyart = ARTNet()
    pyart.train(X, target, trnPar)
    #pyart.train(X, OneClassTarget, trnPar)
    #pyart.recolor(X[trnPar['itrn']], T)
    ##########################################################################################
    ## Evaluate performance
    print '\tKnown-classes SP: %.3f'%pyart.trninfo['val_sp']
    ##########################################################################################
    ## Evaluate novelty detection performance
    prediction = pyart.classify(Xnov)[2]
    perf = (prediction < 0).sum() / float(prediction.shape[0])
    print '\tNovelty-class detection: %.3f'%perf
    pyart.trninfo['Novelty'] = perf

    results['ARTModel'][npcd+1] = pyart.to_save()
##########################################################################################
## Save
#joblib.dump(results, fsave, compress=9)


# end of file


# end of file

