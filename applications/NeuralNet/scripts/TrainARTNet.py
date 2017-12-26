
import numpy as np
import neuralnet as nn
import PyNN.ARTNet as ART
import time
import matplotlib.pyplot as plt
from sklearn import preprocessing

N = 1000

colors = np.array(['b','r','g','y'])
###############################################################################
## MAKE DATA
C1 = np.random.rand(N, 2)*2 +1
C2 = np.random.rand(N, 2)*4 -4
C3 = np.random.rand(N, 2)*2

C2[:,0] = C2[:,0]*20

data = np.concatenate((C1,C2,C3))
target = np.concatenate((np.zeros((N,1)), np.ones((N,1)), 2*np.ones((N,1))))
OneClassTarget = np.zeros((len(data),1))
classes = ['C0','C1','C2']
###############################################################################
## Training Indexes
indexes = np.arange(data.shape[0])
np.random.shuffle(indexes)
ntrn = 0.70
itrn = indexes[:int(data.shape[0]*ntrn)]
ival = indexes[int(data.shape[0]*(1-ntrn)):]
itst = ival



pyart = ART.ARTNet()

trnPar = {
    'itrn': itrn,
    'ival': ival,
    'itst': itst,
    'art_sim_func' : 'euclidian',
    'trn_max_it': 500,
    'trn_eta' : 0.1,
    'trn_nshow' : 100000,
    'trn_max_neurons_rate' : 0.7,
    'trn_phase1' : True,
    'trn_phase2' : True,
    'trn_phase3' : False,
    'trn_initial_radius' : 0,
    'trn_radius_factor' : 1,
    'trn_opt_radius_strategy' : 'std',
    'trn_data_scaler' : None #preprocessing.StandardScaler()
}


if False:
	pyart.train(data, OneClassTarget, trnPar)
	pyart.recolor(data, target)
else:
	pyart.train(data, target, trnPar)




raise Exception("STOP")


pyart.initialize()

Yart = pyart.classify(data)

data = data if pyart.scaler is None else pyart.scaler.transform(data)

###############################################################################
# Truth
plt.figure(figsize=(8,8))
for icls, cls in enumerate(classes):
    #if icls != 1: continue
    idx = np.intersect1d(itrn, np.nonzero(target == icls)[0])
    #idx = np.nonzero(target == icls)[0]    
    plt.plot(data[idx, 0], data[idx, 1], '.', color=colors[icls], alpha=1)
for ineuron in range(pyart.neurons.shape[0]):
    clsnum = int(pyart.classes[ineuron])
    c = colors[clsnum] if clsnum != -1 else 'k'
    plt_neuron = plt.Circle(pyart.neurons[ineuron],pyart.radius[ineuron],color=c,fill=False)
    plt.gca().add_artist(plt_neuron)
plt.axis('equal')
plt.title('Truth')


print pyart.trninfo

raise Exception("STOP")

# Simulation
plt.figure(figsize=(8,8))
for cls in np.unique(Yart[2]):
    idx = np.intersect1d(itrn, np.nonzero(Yart[2] == cls)[0])
    c = 'k' if cls == -1 else colors[cls]
    plt.plot(data[idx, 0], data[idx, 1], '.', color=c, alpha=0.1)

for ineuron in range(pyart.neurons.shape[0]):
    clsnum = int(pyart.classes[ineuron])
    c = colors[clsnum] if clsnum != -1 else 'k'
    plt_neuron = plt.Circle(pyart.neurons[ineuron],
                            pyart.radius[ineuron],color=c,fill=False)
    plt.gca().add_artist(plt_neuron)
plt.axis('equal')
plt.title('Classification')


plt.plot([pyart.neurons[0][0], pyart.neurons[0][0]],
         [pyart.neurons[0][1], pyart.neurons[0][1]+pyart.radius[0]])





###############################################################################


neurons = nn.MatrixF()
classes = nn.RowI()
radius  = nn.RowF()
centroids = nn.MatrixF()
neuron_hits = nn.RowF()
neuron_class_freq = nn.MatrixF()
neuron_class_hits = nn.MatrixF()

artnet.get_neurons(neurons)
artnet.get_classes(classes)
artnet.get_radius(radius)
artnet.get_centroids(centroids)
artnet.get_neuron_hits(neuron_hits)
artnet.get_neuron_class_freq(neuron_class_freq)
artnet.get_neuron_class_hits(neuron_class_hits)

neurons = np.array(neurons)
classes = np.array(classes)
centroids = np.array(centroids)
neuron_hits = np.array(neuron_hits)
neuron_class_freq = np.array(neuron_class_freq)
neuron_class_hits = np.array(neuron_class_hits)

data = np.array(data)
target = np.array(target)
X = data[itrn]
T = target[itrn]
iwinner = nn.RowI()
neuron_out = nn.RowF()
artnet.feedforward(nn.MatrixF(X), neuron_out, iwinner)
iwinner = np.array(iwinner)
neuron_out = np.array(neuron_out)





###############################################################################
## PLOT DATA
import matplotlib.pyplot as plt
colors = ['r','k','y']
if True:
    plt.figure(figsize=(6,6))
    #plt.plot(C1[:,0], C1[:,1], 'x'+colors[0], label='C0')
    #plt.plot(C2[:,0], C2[:,1], 'x'+colors[1], label='C1')
    #plt.plot(C3[:,0], C3[:,1], 'x'+colors[2], label='C2')
    plt.plot(data[:,0], data[:,1], 'x'+colors[0], label='C0')
    ###############################################################################
    ## PLOT NEURONS AND RADIUS
    for ineuron in range(artnet.getNumberOfNeurons()):
        cls = classes[ineuron]
        c = colors[int(cls)] if cls != -1 else 'b'
        plt_neuron = plt.Circle(neurons[ineuron],radius[ineuron],color=c,fill=False)
        plt.gca().add_artist(plt_neuron)
    ## PLOT CENTROIDS
    plt.plot(centroids[0,0], centroids[0,1], 's'+colors[0], ms=10)
    #plt.plot(centroids[1,0], centroids[1,1], 's'+colors[1], ms=10)
    #plt.plot(centroids[2,0], centroids[2,1], 's'+colors[2], ms=10)
    ## PLOT CLASS REAL CENTER
    plt.plot(np.mean(data[:,0]), np.mean(data[:,1]), 'd'+colors[0], ms=10)
    #plt.plot(np.mean(C1[:,0]), np.mean(C1[:,1]), 'd'+colors[0], ms=10)
    #plt.plot(np.mean(C2[:,0]), np.mean(C2[:,1]), 'd'+colors[1], ms=10)
    #plt.plot(np.mean(C3[:,0]), np.mean(C3[:,1]), 'd'+colors[2], ms=10)
    plt.legend(loc='best', ncol = 1, numpoints = 1)
    plt.tight_layout()
    plt.show()



raise Exception("STOP")


###############################################################################
## IO MANAGER
data = nn.MatrixF(data)
target = nn.MatrixF(target)
OneClassTarget = nn.MatrixF(np.zeros((len(data),1)))

# IO manager
iomgr = nn.IOMgr()
if not iomgr.initialize(data, OneClassTarget):
    raise Exception('IOMgr: error loading data to IO Manager')
iomgr.set_trn(nn.RowUI(itrn))
iomgr.set_tst(nn.RowUI(itst))
iomgr.set_val(nn.RowUI(ival))
###############################################################################
## CREATE ART
artnet = nn.ARTNet("euclidian")
###############################################################################
## CREATE ART TRAINING OBJECT
trnmgr = nn.Trainart()

trnmgr.trn_max_it = 100
trnmgr.trn_eta = 0.1
trnmgr.trn_nshow = 10000
trnmgr.trn_max_neurons_rate = 0.5
trnmgr.trn_phase1 = True
trnmgr.trn_phase2 = True
trnmgr.trn_phase3 = False
trnmgr.trn_initial_radius = 0
trnmgr.trn_radius_factor = 0.5
trnmgr.trn_opt_radius_strategy = "std"

trnmgr.set_art(artnet)
trnmgr.set_iomgr(iomgr)

if not trnmgr.initialize():
    raise Exception("ERROR in train manager initialization")

###############################################################################
## TRAIN ART
t0 = time.time()
trnmgr.train()
print 'Training took %.3f s'%(time.time() - t0)

neurons = nn.MatrixF()
classes = nn.RowI()
radius  = nn.RowF()
centroids = nn.MatrixF()
neuron_hits = nn.RowF()
neuron_class_freq = nn.MatrixF()
neuron_class_hits = nn.MatrixF()

artnet.get_neurons(neurons)
artnet.get_classes(classes)
artnet.get_radius(radius)
artnet.get_centroids(centroids)
artnet.get_neuron_hits(neuron_hits)
artnet.get_neuron_class_freq(neuron_class_freq)
artnet.get_neuron_class_hits(neuron_class_hits)

neurons = np.array(neurons)
classes = np.array(classes)
centroids = np.array(centroids)
neuron_hits = np.array(neuron_hits)
neuron_class_freq = np.array(neuron_class_freq)
neuron_class_hits = np.array(neuron_class_hits)

data = np.array(data)
target = np.array(target)
X = data[itrn]
T = target[itrn]
iwinner = nn.RowI()
neuron_out = nn.RowF()
artnet.feedforward(nn.MatrixF(X), neuron_out, iwinner)
iwinner = np.array(iwinner)
neuron_out = np.array(neuron_out)



###############################################################################
## PLOT DATA
import matplotlib.pyplot as plt
colors = ['r','k','y']
if True:
    plt.figure(figsize=(6,6))
    #plt.plot(C1[:,0], C1[:,1], 'x'+colors[0], label='C0')
    #plt.plot(C2[:,0], C2[:,1], 'x'+colors[1], label='C1')
    #plt.plot(C3[:,0], C3[:,1], 'x'+colors[2], label='C2')
    plt.plot(data[:,0], data[:,1], 'x'+colors[0], label='C0')
    ###############################################################################
    ## PLOT NEURONS AND RADIUS
    for ineuron in range(artnet.getNumberOfNeurons()):
        cls = classes[ineuron]
        c = colors[int(cls)] if cls != -1 else 'b'
        plt_neuron = plt.Circle(neurons[ineuron],radius[ineuron],color=c,fill=False)
        plt.gca().add_artist(plt_neuron)
    ## PLOT CENTROIDS
    plt.plot(centroids[0,0], centroids[0,1], 's'+colors[0], ms=10)
    #plt.plot(centroids[1,0], centroids[1,1], 's'+colors[1], ms=10)
    #plt.plot(centroids[2,0], centroids[2,1], 's'+colors[2], ms=10)
    ## PLOT CLASS REAL CENTER
    plt.plot(np.mean(data[:,0]), np.mean(data[:,1]), 'd'+colors[0], ms=10)
    #plt.plot(np.mean(C1[:,0]), np.mean(C1[:,1]), 'd'+colors[0], ms=10)
    #plt.plot(np.mean(C2[:,0]), np.mean(C2[:,1]), 'd'+colors[1], ms=10)
    #plt.plot(np.mean(C3[:,0]), np.mean(C3[:,1]), 'd'+colors[2], ms=10)
    plt.legend(loc='best', ncol = 1, numpoints = 1)
    plt.tight_layout()
    plt.show()


raise Exception("STOP");

###############################################################################
## ART CONSIDERING THE STRATEGY: MAP + NEURON COLORING
trnmgr.recolor(nn.MatrixF(X), nn.MatrixF(T))

neurons = nn.MatrixF()
classes = nn.RowI()
radius  = nn.RowF()
centroids = nn.MatrixF()
neuron_hits = nn.RowF()
neuron_class_freq = nn.MatrixF()
neuron_class_hits = nn.MatrixF()

artnet.get_neurons(neurons)
artnet.get_classes(classes)
artnet.get_radius(radius)
artnet.get_centroids(centroids)
artnet.get_neuron_hits(neuron_hits)
artnet.get_neuron_class_freq(neuron_class_freq)
artnet.get_neuron_class_hits(neuron_class_hits)

neurons = np.array(neurons)
classes = np.array(classes)
centroids = np.array(centroids)
neuron_hits = np.array(neuron_hits)
neuron_class_freq = np.array(neuron_class_freq)
neuron_class_hits = np.array(neuron_class_hits)


plt.figure(figsize=(6,6))

plt.plot(X[classes[iwinner] == 0, 0], X[classes[iwinner] == 0, 1], 'x'+colors[0], label='P0')
plt.plot(X[classes[iwinner] == 1, 0], X[classes[iwinner] == 1, 1], 'x'+colors[1], label='P1')
plt.plot(X[classes[iwinner] == 2, 0], X[classes[iwinner] == 2, 1], 'x'+colors[2], label='P2')

plt.plot(X[neuron_out < 0, 0], X[neuron_out < 0, 1], 'xb', label='?')
###############################################################################
## PLOT NEURONS AND RADIUS
for ineuron in range(artnet.getNumberOfNeurons()):
    cls = classes[ineuron]
    c = colors[int(cls)] if cls != -1 else 'b'
    plt_neuron = plt.Circle(neurons[ineuron],radius[ineuron],color=c,fill=False)
    plt.gca().add_artist(plt_neuron)
## PLOT CENTROIDS
plt.plot(centroids[0,0], centroids[0,1], 's'+colors[0], ms=10)
plt.plot(centroids[1,0], centroids[1,1], 's'+colors[1], ms=10)
plt.plot(centroids[2,0], centroids[2,1], 's'+colors[2], ms=10)
## PLOT CLASS REAL CENTER
plt.plot(np.mean(C1[:,0]), np.mean(C1[:,1]), 'd'+colors[0], ms=10)
plt.plot(np.mean(C2[:,0]), np.mean(C2[:,1]), 'd'+colors[1], ms=10)
plt.plot(np.mean(C3[:,0]), np.mean(C3[:,1]), 'd'+colors[2], ms=10)
plt.legend(loc='best', ncol = 1, numpoints = 1)
plt.tight_layout()
plt.show()




# end of file

