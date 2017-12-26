


import PyNN.ART as PyART
import numpy as np


N = 100
###############################################################################
## MAKE DATA
C1 = np.random.rand(N, 2)*2 +1
C2 = np.random.rand(N, 2)*4 -4
C3 = np.random.rand(N, 2)*2
X = np.concatenate((C1,C2,C3))
T = np.concatenate((np.zeros(N), np.ones(N), 2*np.ones(N)))
###############################################################################
## Training Indexes
indexes = np.arange(X.shape[0])
np.random.shuffle(indexes)
itrn = indexes[:int(X.shape[0]*0.5)]
ival = indexes[int(X.shape[0]*0.5):]
itst = ival
###############################################################################
## Training Parameters
trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'trn_initial_radius' : 0.7,
    'trn_radius_factor'  : 2,
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
###############################################################################
## CREATE ART
nnet = PyART.ART()
###############################################################################
## TRAIN ART
nnet.train(X, T, trnPar)

Y = nnet.classify(X)

#raise Exception('STOP')
###############################################################################
## PLOT DATA
import matplotlib.pyplot as plt
colors = ['r','k','y']
plt.figure(figsize=(6,6))
plt.plot(C1[:,0], C1[:,1], 'x'+colors[0])
plt.plot(C2[:,0], C2[:,1], 'x'+colors[1])
plt.plot(C3[:,0], C3[:,1], 'x'+colors[2])
###############################################################################
## PLOT NEURONS AND RADIUS
for ineuron in range(nnet.neurons.shape[0]):
    icls = int(nnet.classes[ineuron])
    c = colors[icls] if icls != -1 else 'b'
    plt_neuron = plt.Circle(nnet.neurons[ineuron],nnet.radius[ineuron],color=c,fill=False)
    #plt.plot(nnet.neurons[ineuron][0], nnet.neurons[ineuron][1], 'o')
    plt.gca().add_artist(plt_neuron)
plt.tight_layout()
plt.show()
# end of file

