

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.NeuralNet as PyNNet
import PyNN.Initialization as PyInit
import PyNN.DataNorm as PyNorm
from sklearn import preprocessing

from PyNN.ARTNet import ARTNet as ARTNet

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
fcrossv = savedir + 'cvindexes_'+ novcls + '_1024nfft.jbl'
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/ART/'
fsave = savedir + 'art_pcdcons_' + novcls + '_1024nfft_icv%i.jbl'%cvidx
##########################################################################################
## Load PCD File
pcddir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD/Constructive/'
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
known_targets = np.zeros(data.shape[0])
###############################################################################
## DATA INDEXING
indexes = joblib.load(fcrossv)
###############################################################################
## Training Parameters
ninit = 1
trnPar = {
    'itrn': indexes['Indexes'][cvidx]['ITrn'],
    'ival': indexes['Indexes'][cvidx]['IVal'],
    'itst': indexes['Indexes'][cvidx]['ITst'],
    'trn_phase1'             : True,
    'trn_phase2'             : True,
    'trn_initial_radius'     : 0,     # Neuron initial radius
    'trn_max_radius_sample'  : 1 ,   # Number of samples to estimate radius
    'trn_radius_factor'      : 0.8 ,  # Factor applied to the initial radius
    'trn_sim_function'       : 'euclidean',
    'trn_eta'                : 0.1 ,  # Neuron update constant
    'trn_eta_decay'          : 0.1, # Regularization factor
    'trn_nshow'              : 0 ,    # Number of iterations to show
    'trn_tol'                : 1e-1 , # Min allowed tolerance to consider the stall
    'trn_max_stall'          : 4 ,   # Max allowed iterations on stall
    'trn_max_no_new_neuron'  : 6 ,   # Max allowed iterations with no neuron creation
    'trn_max_neurons_rate'   : 0.7 ,  # Percentage of neurons wrt data events
    'trn_max_it'             : 30 ,  # Max number of iterations
    'trn_data_norm' : preprocessing.StandardScaler()
} 


##########################################################################################
## Replicate small class
if False:
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
results['artPar'] = {} # PCD:
results['ARTModel'] = {} # PCD: ART
results['artPar']['Performance'] = {}
results['artPar']['NoveltyPerf'] = {}
##########################################################################################
## LOOP OVER PCDs
# TODO: Several different initializations.
#       Which to save as the best? SP not ok if it is novelty.
#       Greater coverage also might not be the best for novelty...
#for npcd in range(len(PCD['PCDModel'].results[cvidx].PCDNets)):
for npcd in range(10,11):
    print '==> Training for ', npcd+1, ' PCDs'
    #############################################################################
    ## DATA PROJECTION ONTO PCDs
    PCDNet = PCD['PCDModel'].results[cvidx].PCDNets[npcd]
    PCDAvg = PCDNet.avgIn
    PCDStd = PCDNet.stdIn
    PCDMat = PCDNet.W[0]
    X = PCDMat.dot(data.transpose()).transpose()
    Xnov = PCDMat.dot(data_nov.transpose()).transpose()
    T = np.argmax(target, axis=1)
    OneClassTarget = np.zeros(T.shape)
    class_names = PCD['Classes']
    #############################################################################
    # TRAIN
    pyart = None
    arteff = 1e6 # Measured as the unmapped rate
    # Initializations    
    for iinit in range(ninit):
        art = ARTNet(trnPar)
        art.train(X, T, trnPar)
        art.performance(X[trnPar['ival']], T[trnPar['ival']])

        aux = art.outputs(X)

        if art.trn_info['Unmapped'] < arteff:
            arteff = art.trn_info['Unmapped']
            pyart = art
    ##########################################################################################
    ## Final performance
    print '=='*10, ' FINAL ', '=='*10
    print 'Known-classes detection: %.2f'%(pyart.trn_info['SP'])
    for icls, cls in enumerate(np.unique(T)):
        print '\t', class_names[icls], ': %.2f'%(pyart.trn_info['PD_c%i'%cls])
    print '\tUnmapped: %.2f'%(pyart.trn_info['Unmapped'])
    ##########################################################################################
    ## Evaluate novelty detection performance
    classes = pyart.classify(Xnov)[0]
    noveff = (classes == -1).sum() / float(classes.shape[0])
    print 'Novelty detection: %.2f'%(100*noveff)
    pyart.trn_info['Novelty'] = 100*noveff
    rates = np.zeros(np.unique(T).shape[0])
    for icls, cls in enumerate(np.unique(T)):
        idx = classes == cls
        rates[icls] = (idx.sum()) / float(classes.shape[0]) * 100
        print '\t', class_names[icls], ': %.2f'%rates[icls] 
    pyart.trn_info['Nov_false'] = rates
    results['ARTModel'][npcd+1] = pyart

##########################################################################################
## Save
#joblib.dump(results, fsave, compress=9)


# end of file

