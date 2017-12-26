
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas
from sklearn.externals import joblib
from sklearn import metrics
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
import scipy as sc
import PyNN.ARTNet as pyart
from sklearn import preprocessing
import matplotlib as mpl



np.set_printoptions(precision=2, suppress=True)

# Standard styles for each class
dashes = np.array([[],[10,10],[10,4,2,4],[10,5,100,5]] )
colors = np.array(['b','r','g','y'])
markers= np.array(['s','o','d','^'])


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['legend.handlelength'] = 3
mpl.rcParams['legend.borderpad'] = 0.3
mpl.rcParams['legend.numpoints'] = 1

if len(sys.argv) != 4:
    print 'Usage:'
    print '$ ', sys.argv[0], ' <novelty> <cv index> <npcd>'
    sys.exit(-1)
novcls = sys.argv[1]
icv = int(sys.argv[2])
npcd = int(sys.argv[3])


classes = np.array(['ClasseA','ClasseB','ClasseC','ClasseD'])
sonarhome = os.getenv('SONARHOME')
sonarnov = os.getenv('SONARNOVELTY') + '/NEW/'
nfft = 1024
nPts = 400

###############################################################################
## LOAD DATA
data = {}
target = {}
data_nov = {}
cvPar = {}

fdata = sonarhome + '/data/novelty_' + novcls + '_' + str(nfft) + 'nfft.jbl'
obj = joblib.load(fdata)
data[novcls] = obj['data']
data_nov[novcls] = obj['data_nov']
target[novcls] = obj['target']
cvPar[novcls] = obj['cvPar']

###############################################################################
## LOAD PCD
pcdnet = {}
pcdnet[novcls] = {}
# Deflation (or independent)
filepcd = sonarnov + '/PCD/pcdind_' + novcls + '_' + str(nfft) + 'nfft.jbl'
pcdnet[novcls]['def'] = joblib.load(filepcd)
# Cooperative (or constructive)
filepcd = sonarnov + '/PCD/pcdcons_' + novcls + '_' + str(nfft) + 'nfft.jbl'
pcdnet[novcls]['coo'] = joblib.load(filepcd)

###############################################################################
## PREPARE DATA
Wdef = pcdnet[novcls]['def']['PCDModel'].results[icv].PCDNets[npcd-1].W[0]
Wcoo = pcdnet[novcls]['coo']['PCDModel'].results[icv].PCDNets[npcd-1].W[0]
T = target[novcls]
X = data[novcls]
Xnov = data_nov[novcls]
Xpcd_def = pandas.DataFrame(Wdef.dot(X.transpose()).transpose())
XnovPcd_def = pandas.DataFrame(Wdef.dot(Xnov.transpose()).transpose())
Xpcd_coo = pandas.DataFrame(Wcoo.dot(X.transpose()).transpose())
XnovPcd_coo = pandas.DataFrame(Wcoo.dot(Xnov.transpose()).transpose())
###############################################################################
## TRAINING PARAMETERS
artPar = {
    'trn_initial_radius' : 5,
    'trn_max_radius_sample': 5000,
    'trn_radius_factor'  : 1,
    'trn_phase_flags': np.array([True, True, False]),
    #'trn_similarity_function': 'euclidean',
    'trn_nproc': 1, # it does not seem to be faster if != 1
    'trn_eta': 0.1,
    'trn_eta_decay': 0.01,
    'trn_mem_n_it': 2,   # number of iterations to calculate memory factor
    'trn_nshow': 0,
    'trn_tol': 1e-5,
    'trn_max_it' : 100,
    'trn_max_stall': 10, # max number of iterations without neuron radius update
    'trn_max_no_new_neuron' : 10,
    '': preprocessing.StandardScaler()
}
# Other Train paramters
trnPar = {
    'itrn': cvPar[novcls]['Indexes'][icv]['ITrn'],
    'ival': cvPar[novcls]['Indexes'][icv]['IVal'],
    'itst': cvPar[novcls]['Indexes'][icv]['ITst']
}
###############################################################################
## TRAIN ART NET
dim1 = 7
dim2 = 23

idx = np.arange(Xpcd_coo.shape[0])
np.random.shuffle(idx)
art_data = Xpcd_coo.values[:, [dim1, dim2]]
art_target = np.argmax(T, axis=1)
art_coo = pyart.ARTNet(artPar)
OneClassTarget = np.zeros(art_data.shape[0])
art_coo.train(art_data, OneClassTarget, trnPar, ['C0'])
art_coo.recolor_neurons(art_data, art_target, trnPar, phase3 = True)



###############################################################################
## EVALUATE
#That = nnet.classify(X)
#CMtrn = confusion_matrix(T[itrn], That[itrn])
#CMtst = confusion_matrix(T[itst], That[itst])

###############################################################################
## PLOT IN 2D
#plt.figure(figsize=(6,6))
#for icls in np.unique(T):
#    plt.plot(X[T==icls,0], X[T==icls,1], '.')
#for ineuron in range(nnet.neurons.shape[0]):
#    icls = int(nnet.classes[ineuron])
#    plt_neuron = plt.Circle(nnet.neurons[ineuron],nnet.radius[ineuron],color='k',fill=False)
#    plt.plot(nnet.neurons[ineuron][0], nnet.neurons[ineuron][1], 'ok')
#    plt.gca().add_artist(plt_neuron)
#plt.tight_layout()
# end of file








