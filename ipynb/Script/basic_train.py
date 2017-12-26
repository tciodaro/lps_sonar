

import numpy as np
import os
# Network import: should be in the path if framework configured correctly
import neuralnet as nn
import PyNN.NeuralNet as PyNNet
import PyNN.TrnInfo as PyTrnInfo
from sklearn.externals import joblib
import time

##########################################################################################
## CONFIGURATION
#fname = 'lofar_data_1024fft_3dec.jbl'
fname = 'lofar_data_1024fft_3dec_fromMat.jbl'
trnPerc = 0.6   # Global training parameters
valPerc = 0.2
classes = ['ClasseA','ClasseC','ClasseD']
nPts = 400
nEvts = -1

netBuilder = nn.RProp # network type
lrn_rate    = 0.01
inc_eta     = 1.4;
dec_eta     = 0.3;

nepochs = 400
nNeurons = 1
nshow = 40
initup = 1.0
initlo = -1.0
nInit = 1 # random initializations
perfType = 'SP'
savedir = '/tmp/'
results = {}
##########################################################################################
## SAVE FILE
fsave = savedir + 'net_'+ ''.join(classes) + '_' + perfType + '.jbl'
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
for ship in data.keys():
    if ship not in classes:
        data.pop(ship)
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
    for runData in data[ship].values():
        idx = np.arange(0, runData['Signal'].shape[1]) + offset
        np.random.shuffle(idx)
        ntrn = int(idx.shape[0] * trnPerc) + 1
        nval = int(idx.shape[0] * valPerc) + 1
        itrn += idx[:ntrn].tolist()
        ival += idx[ntrn:ntrn+nval].tolist()
        itst += idx[ntrn+nval:].tolist()
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
X = X.transpose() # Transpose X, so events are rows (T is already correct)
if not len(itst):
    itst = ival
##########################################################################################
## CREATE NN IOMANAGER
Xarray = X
Tarray = T
X = nn.MatrixF(X.tolist())
T = nn.MatrixF(T.tolist())
# Network IOMgr
iomgr = nn.PyIOMgr_Pattern()
iomgr.index_strategy = 'replicate'
if not iomgr.load(X, T):
    raise Exception('ERROR LOADING DATA TO IOMGR')
iomgr.set_trn(nn.RowUI(itrn))
iomgr.set_tst(nn.RowUI(itst))
iomgr.set_val(nn.RowUI(ival))
iomgr.mapstd(); # NN normalization

##########################################################################################
## CREATE TRAIN MANAGER AND TRAIN PCD
trnmgr = nn.Trainbp()
trnmgr.set_iomgr(iomgr);
trnmgr.net_task = 'classification'
trnmgr.nepochs = nepochs
trnmgr.nshow = nshow
trnmgr.fbatch = True
# Configuration
funcs = 'tanh:tanh'

results['NInit'] = nInit

currPerf = 0.0;
nodes = str(iomgr.in_dim) + ":"+ str(nNeurons) + ":" + str(iomgr.out_dim)
nnet = netBuilder(nodes, funcs)
nnet.lrn_rate = lrn_rate
nnet.inc_eta = inc_eta
nnet.dec_eta = dec_eta


nnet.initialize(initup, initlo) #network must be initialized before the train manager

t0 = time.time()
# Over initializations
tempnet = nnet.copy()
initPerf = -999.0
trnmgr.initialize()
trninfo = nn.TrnInfo_Pattern(trnmgr.get_trninfo())
for iinit in range(nInit):
    print '\t\tInitialization #', iinit,
    #tempnet.init_weights(initup, initlo)
    PyNNet.NeuralNet.initnw(tempnet, Xarray)
    trnmgr.set_net(tempnet)
    trnmgr.train()
    print 'performance: ', trnmgr.get_trninfo().performance(perfType)
    if initPerf < trnmgr.get_trninfo().performance(perfType):
        initPerf = trnmgr.get_trninfo().performance(perfType)
        trninfo.copy(trnmgr.get_trninfo())
        nnet.copy(tempnet)
currPerf = trninfo.performance(perfType)
# Get best net and compare with previous number of PCDs
pytrn = PyTrnInfo.TrnInfo(trninfo)
pytrn.itrn = np.array(itrn)
pytrn.itst = np.array(itst)
pytrn.ival = np.array(ival)
pytrn.target = np.array(T, 'f')
pynn = PyNNet.NeuralNet(nnet)
pynn.trn_info = pytrn
pynn.avgIn = np.array(iomgr.get_mean(), 'f')
pynn.stdIn = np.array(iomgr.get_std(), 'f')
print 'Final performance: ', currPerf,
print ' (time: %2.2fs)'%(time.time()-t0)




# end of file







