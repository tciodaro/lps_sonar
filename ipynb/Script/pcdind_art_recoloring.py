

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
from sklearn import preprocessing

import PyNN.ART as PyART

from sklearn.externals import joblib
import time

#if len(sys.argv) != 3:
#    print 'Missing arguments to script:'
#    print '> ', sys.argv[0], ' <fname>'
#    sys.exit(-1)
##########################################################################################
## CONFIGURATION
fdata = 'lofar_data_1024fft_3dec_fromMat.jbl'
filepcd = os.getenv('SONARHOME') + '/' + sys.argv[1]
fname = os.getenv('SONARHOME') + '/' + sys.argv[2]
#filepcd = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_Independent/pcdind_cv_ClasseA_fromMat_1024nfft.jbl'
#fname = os.getenv('SONARHOME') + '/results/classification/novelty/PCD04_ARTIndependent/art_pcdind4_cv_ClasseA_fromMat_1024nfft.jbl'
nPts = 400
nEvts = -1
##########################################################################################
## LOAD ART FILE
artobj = joblib.load(fname)
nov_classes = artobj['Novelties']
npcd = artobj['ARTModel'].results[0][0].values()[0].neurons.shape[1]
##########################################################################################
## DATA LOADING
fdata = os.getenv('SONARHOME') + '/data/' + fdata
# Load and filter data
rawdata = joblib.load(fdata)
for shipData in rawdata.values():
    for runData in shipData.values():
        runData['Signal'] = runData['Signal'][:nPts,:] if nEvts == -1 else runData['Signal'][:nPts,:nEvts]
        runData['Freqs'] = runData['Freqs'][:nPts]
# Pop classes not used
nov_rawdata = {}
for ship in rawdata.keys():
    if ship in nov_classes:
        nov_rawdata[ship] = rawdata.pop(ship)
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
target = np.array(np.argmax(target,axis=1), 'i')
##########################################################################################
## LOAD PCD MODEL
pcdobj = joblib.load(filepcd)
##########################################################################################
## LOOP OVER CROSS-VALIDATION
NCV = artobj['cvPar']['CVNSel']
radius = np.sort(artobj['ARTModel'].results[0][0].keys())
effs= np.zeros((NCV, len(radius)))
for icv in range(NCV):
    PCDMat = pcdobj['PCDModel'].results[icv].PCDNets[npcd-1].W[0]
    #############################################################################
    ## DATA PROJECTION ONTO PCDs
    X = PCDMat.dot(data.transpose()).transpose()
    Xnov = PCDMat.dot(data_nov.transpose()).transpose()
    #############################################################################
    ## LOOP OVER ART WITH DIFFERENT RADIUS
    for irad, rad in enumerate(radius):
        art = artobj['ARTModel'].results[icv][0][rad]
        Xart = art.trn_data_norm.transform(X)
        Xnov_art = art.trn_data_norm.transform(Xnov)
        #########################################################################
        ## RECOLOR THE ART NEURONS
        art._associate_class(Xart, target)
        #########################################################################
        ## PRINT CLASSIFICATION
        print '='*20, ' Radius: ', art.trn_initial_radius
        Y = art.classify(Xart)
        pd = [0,0,0]
        for c in range(3):
            tpr = float((Y[1][target==c] == c).sum()) / (target == c).sum()
            pd[c] = tpr
            print '\tClass ', c, ' ', tpr
        effs[icv, irad] = np.mean(pd)
        # ACCURACY
        itst = art.trn_info.itst
        Ynov = art.classify(Xnov_art)
        print '\t\tAcc.: ', float((Y[1][itst] != -1).sum()) / itst.shape[0]
        print '\t\tNov.: ', float((Ynov[1] == -1).sum()) / Ynov[1].shape[0]
        print '\t\tTrn.: ', art.trn_info.perf

    #break
#raise Exception('STOP')

joblib.dump(artobj, fname, compress=9)


# end of file

