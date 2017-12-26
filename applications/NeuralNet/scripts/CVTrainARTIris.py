import PyNN.ART as PyART
import numpy as np
from sklearn.datasets import load_iris
import PyNN.CrossValidation as PyCV


# Get Data
iris = load_iris()
data = iris['data']
target = iris['target']
indexes = np.arange(data.shape[0])
known_targets = np.zeros(data.shape[0])
###############################################################################
## Training Parameters
trnPar = {
    'trn_initial_radius' : -999,
    'trn_radius_factor'  : 1,
    'trn_radius_metric' : 'euclidean',
    'trn_similarity_function': None,
    'trn_eta': 0.1,
    'trn_eta_min': 0.1*np.exp(-5),
    'trn_mem_n_it': 2,   # number of iterations to calculate memory factor
    'trn_mem_factor': 1,
    'trn_nshow': 1,
    'trn_tol': 1e-6,
    'trn_max_stall': 10, # max number of iterations without neuron radius update
    'trn_max_no_new': 5
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
pycv = PyCV.CVFold(cvPar)
pycv.train(data, known_targets, PyART.ART, {}, trnPar)
##########################################################################################
## Evaluate performance
perfs = np.zeros(cvPar['CVNSel'])
print 'Performances:'
for icv, nnet in enumerate(pycv.results):
    print '\tCV ', icv, ': ', nnet.trn_info.perf
    perfs[icv] = nnet.trn_info.perf
print 'Final: %.3f +- %.3f'%(np.mean(perfs), np.std(perfs)),

raise Exception('STOP')
###############################################################################
## PLOT DATA
import matplotlib.pyplot as plt
colors = ['r','g','y']
plt.plot(data[target==0,0], data[target==0,1], '.'+colors[0])
plt.plot(data[target==1,0], data[target==1,1], '.'+colors[1])
plt.plot(data[target==2,0], data[target==2,1], '.'+colors[2])
###############################################################################
## PLOT NEURONS AND RADIUS
for ineuron in range(nnet.neurons.shape[0]):
    icls = int(nnet.classes[ineuron])
    c = colors[icls] if icls != -1 else 'b'
    plt_neuron = plt.Circle(nnet.neurons[ineuron],nnet.radius[ineuron],color=c,fill=False)
    plt.gca().add_artist(plt_neuron)
plt.tight_layout()
plt.show()
