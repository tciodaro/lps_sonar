

import numpy as np
import os
import sys
# Network import: should be in the path if framework configured correctly
import neuralnet as nn

import PyNN.CrossValidation as PyCV
import PyNN.ART as PyART

from sklearn.externals import joblib
import time


if len(sys.argv) != 2:
    print 'ERROR: usage'
    print sys.argv[0], ' <novelty>'
    sys.exit(-1)

##########################################################################################
## CONFIGURATION
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
nov_classes = [sys.argv[1]]
full_classes = ['ClasseA','ClasseB','ClasseC', 'ClasseD']
classes = np.sort(np.setdiff1d(full_classes, nov_classes))
novcls = ''.join(nov_classes) # label
nPts = 400
nEvts = -1
npcd = 18
IPCD = {
    'ClasseA': 2,
    'ClasseB': 4,
    'ClasseC': 0,
    'ClasseD': 2
}
results = {}
##########################################################################################
## Load PCD File
pcddir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_Constructive/'
if len(nov_classes):
    filepcd = pcddir + 'pcdcons_cv_'+ novcls + '_fromMat_1024nfft.jbl'
else:
    filepcd = pcddir + 'pcdcons_cv_full_fromMat_1024nfft.jbl'
PCD = joblib.load(filepcd)
PCDNet = PCD['Model'].results[IPCD[novcls]].PCDNets[npcd-1]
PCDAvg = PCDNet.avgIn
PCDStd = PCDNet.stdIn
PCDMat = PCDNet.W[0]
##########################################################################################
## SAVE FILE
savedir = os.getenv('SONARHOME') + '/results/classification/novelty/PCD_ARTNet/'
if len(nov_classes):
    fsave = savedir + 'art_pcdcons_cv_'+ novcls + '_fromMat_1024nfft.jbl'
else:
    fsave = savedir + 'art_pcdcons_cv_full_fromMat_1024nfft.jbl'
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
##########################################################################################
## DATA PROJECTION ONTO PCDs
#data = (data - PCDAvg) / PCDStd
data = PCDMat.dot(data.transpose()).transpose()
#data_nov = (data_nov - PCDAvg) / PCDStd
data_nov = PCDMat.dot(data_nov.transpose()).transpose()
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
###############################################################################
## Training Parameters
trnPar = {
    'trn_initial_radius' : -999,
    'trn_radius_factor'  : 1,
    'trn_radius_metric' : 'euclidean',
    'trn_similarity_function': None,
    'trn_eta':1,
    'trn_eta_min': 0.1*np.exp(-5),
    'trn_mem_n_it': 2,   # number of iterations to calculate memory factor
    'trn_mem_factor': 1,
    'trn_nshow': 1,
    'trn_tol': 1e-6,
    'trn_max_stall': 6, # max number of iterations without neuron radius update
    'trn_max_no_new': 6,
    'trn_max_radius_sample': 1000
}
##########################################################################################
## Cross Validation
cvPar = {
    'indexes': indexes,
    'TrnPerc': 0.7,
    'ValPerc': 0.3,
    'CVNSel' : 1,
    'CVNFold': 10
}
##########################################################################################
## Train Model
gridSearch = {}
gridSearch['Parameters'] = np.linspace(0.6,1.2, 4)
gridSearch['results'] = []
for p in gridSearch['Parameters']:
    trnPar['trn_radius_factor'] = p
    pycv = PyCV.CVMultiFold(cvPar)
    pycv.train(data, known_targets, PyART.ART, {}, trnPar)
    ##########################################################################################
    ## Evaluate performance
    perfs = np.zeros(cvPar['CVNSel'])
    print 'Known-classes performances:'
    for icv, nnet in enumerate(pycv.results):
        print '\tCV ', icv, ': %.3f'%nnet.trn_info.perf
        perfs[icv] = nnet.trn_info.perf
    print 'Final: %.3f +- %.3f'%(np.mean(perfs), np.std(perfs))
    ##########################################################################################
    ## Evaluate novelty detection performance
    perfs = np.zeros(cvPar['CVNSel'])
    print 'Novelty-classes performances:'
    for icv, nnet in enumerate(pycv.results):
        Y = nnet.classify(data_nov)
        perf = np.sum(Y==-1) / float(Y.shape[0])
        print '\tCV ', icv, ': ', perf
        perfs[icv] = perf
    print 'Final: %.3f +- %.3f'%(np.mean(perfs), np.std(perfs))
    results = {}
    results['cvPar'] = cvPar
    results['trnPar'] = trnPar
    results['IPCD'] = IPCD
    results['Model'] = pycv
    gridSearch['results'].append(results)
##########################################################################################
## Save
joblib.dump(gridSearch, fsave, compress=9)



## PLOT
plt.figure(figsize=(6,6))
for icls in np.unique(T):
    plt.plot(X[T==icls,0], X[T==icls,1], '.')
for ineuron in range(pyart.neurons.shape[0]):
    icls = int(pyart.classes[ineuron])
    plt_neuron = plt.Circle(pyart.neurons[ineuron],pyart.radius[ineuron],color='k',fill=False)
    plt.plot(pyart.neurons[ineuron][0], pyart.neurons[ineuron][1], 'ok')
    plt.gca().add_artist(plt_neuron)


# end of file











