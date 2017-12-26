

import numpy as np
import neuralnet as nn
import time
import copy
from . import TrnInfo as PyInfo
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

import sys


class ARTNet(object):
    def __init__(self, par = None):
        self.neurons = None
        self.radius = None
        self.classes = None
        self.neuron_hits = None
        self.neuron_class_freq = None
        self.neuron_class_hits = None
        self.neuron_move_hist = None
        self.neuron_class_rate = None
        # Train parameters
        self.trn_phase2             = True
        self.trn_initial_radius     = 0     # Neuron initial radius
        self.trn_min_radius         = 0.05  # minimum radius
        self.trn_max_radius_sample  = 0    # Number of samples to estimate radius
        self.trn_radius_factor      = 1.0   # Factor applied to the initial radius
        self.trn_sim_function       = 'euclidean'
        self.trn_eta                = 0.1   # Neuron update constant
        self.trn_eta_decay          = 0.001 # Regularization factor
        self.trn_nshow              = 0     # Number of iterations to show
        self.trn_tol                = 1e-6  # Min allowed tolerance to consider the stall
        self.trn_max_stall          = 10    # Max allowed iterations on stall
        self.trn_info               = PyInfo.TrnInfo()
        self.trn_data_norm          = None  # Data normalizer
        self.trn_max_neurons_rate   = 0.7   # Percentage of neurons wrt data events
        self.trn_max_it             = 100   # Max number of iterations
        self.trn_max_no_new_neuron  = 10    # Max number of iterations without neuron creation
        self.trn_phase2_max_eff     = 0.005 # Max allowed inefficiency in phase 2
        self.__D                    = None
        self.__dist_function = None
        self.__used_neurons = 0

        ## Check what is initPar
        if par is not None:
            if isinstance(par, ARTNet):
                for key, val in par.__dict__.items():
                    setattr(self, key, copy.deepcopy(val))
            elif isinstance(par, dict):
                for key, val in par.items():
                    setattr(self, key, copy.deepcopy(val))


    def __print(self, *args):
        cmd = ''.join([str(x) for x in args])
        if sys.version_info[0] < 3:
            print(cmd)
            

        else:
            print(cmd)

    ######################################################################################
    """
        Default similarity function.
        X: input array in the form [nevents, ndim]
        W: neurons, in the form [nneurons, ndim]
        radius: neuron radius, in the form [nneurons]

        Output: array([nevents, nneurons])
    """
    def __similarity_function(self, X, W, radius):
        return radius - self.__dist_function.pairwise(X, W)


    ######################################################################################
    """
        Trains the art network

        - Phase 1: single class contention.
        - Phase 2: fine radius adjustment.
    """
    def train(self, data, target, trnPar, class_labels = None):
        t1 = time.time()
        tgt_classes = np.unique(target)
        nclasses = len(tgt_classes)
        if class_labels is None: class_labels = np.arange(nclasses).astype('str')
        self.__print('Training ARTNet for ', nclasses, ' classes.')
        self.init(data, target, trnPar)
        # Data Normalization
        self.__print('\tData normalization')
        if self.trn_data_norm is not None:
            self.trn_data_norm.fit(data[trnPar['itrn']])
            data = self.trn_data_norm.transform(data)
        # Train Phase 1
        self.__print('\tStarting Phase 1: individual class contention')
        for cls, clsname in zip(tgt_classes, class_labels):
            self.__print('\tClass: ', clsname)
            self.__train_phase1(data, target, trnPar, cls)
        # Remove unused neurons
        used_neurons           =  self.classes != -1
        self.neurons           =  self.neurons[used_neurons]
        self.radius            =  self.radius[used_neurons]
        self.classes           =  self.classes[used_neurons]
        self.neuron_class_freq =  self.neuron_class_freq[used_neurons]
        self.neuron_class_hits =  self.neuron_class_hits[used_neurons]
        self.neuron_hits       =  self.neuron_hits[used_neurons]
        self.neuron_move_hist  = self.neuron_move_hist[used_neurons]
        self.neuron_class_rate = self.neuron_class_rate[used_neurons]
        # Train Phase 2
        if self.trn_phase2:
            self.__print('\tStarting Phase 2: radius optimization')
            for cls, clsname in zip(tgt_classes, class_labels):
                self.__train_phase2(data, target, trnPar,  cls)
        # Finish training
        self.prune(data, target, trnPar) # Remove neurons that never win
        self.recolor(data, target, trnPar, fNormed = True)

    ######################################################################################
    """
        Initialize structures
    """
    def init(self, data, target, trnPar):
        # Initialize structures
        r = self.trn_max_neurons_rate
        (nsamples, ndim) = data.shape
        nclasses = np.unique(target).shape[0]
        ntrn = len(trnPar['itrn'])
        self.__used_neurons = 0
        max_neurons = int(r*ntrn)
        self.neurons    =  np.zeros((max_neurons, ndim)) * np.nan
        self.radius     = -np.ones(max_neurons) * np.nan
        self.classes    = -np.ones(max_neurons)
        self.neuron_class_freq =  np.zeros((max_neurons, nclasses))
        self.neuron_class_hits=  np.zeros((max_neurons, nclasses))
        self.neuron_class_rate=  np.zeros((max_neurons, nclasses))
        self.neuron_hits   =  np.zeros(max_neurons)
        self.neuron_move_hist = np.zeros(max_neurons)
        self.__performance_init(data, target, trnPar)
        if self.trn_sim_function == 'euclidean':
            self.__dist_function = DistanceMetric.get_metric(self.trn_sim_function)
        elif self.trn_sim_function == 'mahalanobis':
            C = np.cov(data[trnPar['itrn']].T)
            self.__dist_function = DistanceMetric.get_metric(self.trn_sim_function, V = C)


    ######################################################################################
    """
        Winner function. Return the most similar neuron.
    """
    def winner(self, data, neurons, radius):
        svalues = self.__similarity_function(data, neurons, radius)
        return (np.max(svalues, axis=1), np.argmax(svalues, axis=1))

    ######################################################################################
    """
        Operates the ARTNet with the data, returning the maximum similarity, most probable
        class and winning neuron for each data sample.
    """
    def operate(self, data):
        max_radius = self.radius.max()
        sim, ineuron = self.winner(data, self.neurons, max_radius*np.ones(self.radius.shape))
        cls = self.classes[ineuron]
        cls[sim < 0] = -1
        return np.array([sim, cls, ineuron])
    ######################################################################################
    """
        Outputs the result per class, considering the neuron for class C closest to the
        data point.
    """        
    def outputs(self, data, fNormed = False):
        if not fNormed and self.trn_data_norm is not None:
            data = self.trn_data_norm.transform(data)

        classes = np.unique(self.classes)
        out = np.zeros((data.shape[0], classes.shape[0])) * np.nan
        rad = np.zeros((data.shape[0], classes.shape[0])) * np.nan
        for icls, cls in enumerate(classes):
            ineurons = self.classes == cls
            neurons = self.neurons[ineurons]
            radius  = self.radius[ineurons]
            sim, iwin = self.winner(data, neurons, radius)
            out[:, icls] = sim
            rad[:, icls] = self.radius[iwin]
        return out, rad
    ######################################################################################
    """
        Train the art network to contain each class
    """
    def __train_phase1(self, data, target, trnPar,  cls):
        t1 = time.time()
        it = 0
        itrn = np.intersect1d(trnPar['itrn'], np.nonzero(target == cls)[0])
        ival = np.intersect1d(trnPar['ival'], np.nonzero(target == cls)[0])
        if not ival.shape[0]: ival = itrn
        itst = np.intersect1d(trnPar['itst'], np.nonzero(target == cls)[0])
        if not itst.shape[0]: itst = ival

        radius_init = self.optimal_radius(data[itrn]) if self.trn_initial_radius == 0 \
                                                      else self.trn_initial_radius
        self.__print('\t\tInitial radius: ', radius_init)
        # Initialize the first neuron
        max_neurons = int(itrn.shape[0] * self.trn_max_neurons_rate)
        beg_neurons = self.__used_neurons
        self.neurons[beg_neurons] = data[itrn[0]]
        self.radius[beg_neurons] = radius_init
        self.classes[beg_neurons] = cls
        nneuron = beg_neurons+1
        stalled_neurons = 0
        no_new_neurons = 0

        self.__print('\t\tMaxNeurons: ', max_neurons)
        self.__print('\t\tTrainingSize: ', itrn.shape[0])

        while True:
            ##################### LOOP OVER ALL SAMPLES
            np.random.shuffle(itrn)
            flag_stalled_neurons = True
            flag_no_new_neuron = True
            for i in itrn:
                x = data[i:i+1]
                neurons = self.neurons[beg_neurons:nneuron]
                radius  = self.radius[beg_neurons:nneuron]
                # Event similarity: most similar only
                sim, ineuron = self.winner(x, neurons, radius)
                ineuron = beg_neurons + ineuron
                # not in any neuron radius. Create new neuron
                if sim < 0 and nneuron-beg_neurons < max_neurons:
                    self.neurons[nneuron] = x
                    self.radius[nneuron] = radius_init  
                    self.classes[nneuron] = cls 
                    nneuron = nneuron + 1
                    flag_stalled_neurons = False
                    flag_no_new_neuron = False
                else: # data is inside neuron walk towards it!
                    dx = (x - self.neurons[ineuron]) * self.trn_eta**self.trn_eta_decay
                    flag_stalled_neurons = flag_stalled_neurons and (np.sqrt(dx.dot(dx.T)) < self.trn_tol)
                    self.neurons[ineuron] = self.neurons[ineuron] + dx
                    self.neuron_move_hist[ineuron] = self.neuron_move_hist[ineuron] + 1
            ############### PERFORMANCE MONITORING
            neurons = self.neurons[beg_neurons:nneuron]
            radius  = self.radius[beg_neurons:nneuron]
            ############### STOP CRITERIA
            it = it + 1
            if it == self.trn_max_it:
                self.__print('\t\t=> Max number of iterations reached! (%i)'%it)
                break
            stalled_neurons = stalled_neurons + int(flag_stalled_neurons)
            if stalled_neurons >= self.trn_max_stall:
                self.__print('\t\t=> Neurons are stalled. Abort!')
                break
            no_new_neurons = no_new_neurons + int(flag_no_new_neuron)
            if no_new_neurons >= self.trn_max_no_new_neuron:
                self.__print('\t\t=> Neurons are not created. Abort!')
                break
        # Resume performance measurements
        self.__used_neurons = nneuron
        neurons = self.neurons[beg_neurons:nneuron]
        radius  = self.radius[beg_neurons:nneuron]
        sim = self.winner(data[itrn], neurons, radius)[0]

        unmapped = (100*(sim < 0).sum() / float(sim.shape[0]))

        self.trn_info['NeuronsCreated_c%i'%cls] = (nneuron - beg_neurons)
        self.trn_info['Unmapped_c%i'%cls] = unmapped
        self.trn_info['MaxNeurons_c%i'%cls] = max_neurons
        self.trn_info['InitRadius_c%i'%cls] = radius_init
        self.trn_info['StalledCount_c%i'%cls] = stalled_neurons
        self.trn_info['ItCount_c%i'%cls] = it

        self.__print('\t\t=> Neurons created: ', (nneuron - beg_neurons))
        self.__print('\t\t=> Unmapped data: %.2f %%'%unmapped)
        self.__print('\tPhase 1 completed in %.2f s'%(time.time()-t1))


    ######################################################################################
    """
        Train the art network to adjust the already trained neurons for each class
    """
    def __train_phase2(self, data, target, trnPar,  cls):
        t1 = time.time()
        itrn = np.intersect1d(trnPar['itrn'], np.nonzero(target == cls)[0])
        ival = np.intersect1d(trnPar['ival'], np.nonzero(target == cls)[0])
        if not ival.shape[0]: ival = itrn
        itst = np.intersect1d(trnPar['itst'], np.nonzero(target == cls)[0])
        if not itst.shape[0]: itst = ival
        idx_neurons = np.nonzero(self.classes == cls)[0]
        neurons = self.neurons[idx_neurons]
        radius  = self.radius[idx_neurons]
        # simulate data
        sim, ineuron = self.winner(data[itrn], neurons, radius)
        # loop over winning neurons
        nneuron = 0
        prev_perf = self.trn_info.metrics['phase1_eff_val_c%i'%cls][-1]
        for iwin in np.unique(ineuron):
            prev_pref = 0.001 if prev_perf == 0.0 else prev_perf
            idx = ineuron == iwin
            if (idx & (sim >= 0)).sum() != 0:
                # get the longgest distance inside the neuron radius
                new_radius = np.max(radius[iwin] - sim[idx & (sim >= 0)])*1.001 # just a bit bigger
                if new_radius < self.trn_min_radius:
                    new_radius = self.trn_min_radius
                self.radius[idx_neurons[iwin]] = new_radius
                nneuron = nneuron + 1
        # Check how many fell inside the neurons
        sim, ineuron = self.winner(data[ival], neurons, self.radius[idx_neurons])
        new_perf = (sim > 0).sum() / float(sim.shape[0])
        self.__print('\t\tAdjusted ', nneuron,' neurons. ')
        self.__print('\t\tDone in %.2f s'%(time.time() - t1))

    ######################################################################################
    def prune(self, data, target, trnPar):
        t1 = time.time()
        if trnPar is None:
            itrn = np.arange(data.shape[0])
        else:
            itrn = trnPar['itrn']
        X = data[itrn]
        T = target[itrn]
        tgt_classes = np.unique(T)
        used_neurons = []
        for ineuron in range(self.neurons.shape[0]):
            sim = self.winner(X, self.neurons[ineuron:ineuron+1],
                                 self.radius[ineuron:ineuron+1])[0]
            if (sim >= 0).sum() != 0:
                used_neurons.append(ineuron)
        pruned = self.neurons.shape[0] - len(used_neurons)
        self.neurons           =  self.neurons[used_neurons]
        self.radius            =  self.radius[used_neurons]
        self.classes           =  self.classes[used_neurons]
        self.neuron_class_freq =  self.neuron_class_freq[used_neurons]
        self.neuron_class_hits =  self.neuron_class_hits[used_neurons]
        self.neuron_hits       =  self.neuron_hits[used_neurons]
        self.neuron_move_hist  = self.neuron_move_hist[used_neurons]
        self.neuron_class_rate = self.neuron_class_rate[used_neurons]

        self.__print('Pruned %i neurons. Done in %i s'%(pruned, time.time()-t1))

    ######################################################################################
    def recolor(self, data, target, trnPar = None, fNormed = False):
        t1 = time.time()
        if trnPar is None:
            itrn = np.arange(data.shape[0])
        else:
            itrn = trnPar['itrn']
        X = data[itrn]
        T = target[itrn]
        if not fNormed and self.trn_data_norm is not None:
            X = self.trn_data_norm.transform(X)

        tgt_classes = np.unique(T)
        # Reset counters
        nclasses = tgt_classes.shape[0]
        nneurons = self.neurons.shape[0]
        self.neuron_class_freq =  np.zeros((nneurons, nclasses))
        self.neuron_class_hits =  np.zeros((nneurons, nclasses))
        self.neuron_class_rate =  np.zeros((nneurons, nclasses))
        for ineuron in range(self.neurons.shape[0]):
            sim = self.winner(X, self.neurons[ineuron:ineuron+1],
                                 self.radius[ineuron:ineuron+1])[0]
            for icls, cls in enumerate(tgt_classes):
                idx = T == cls
                den = float((sim >= 0).sum()) if (sim >= 0).sum() != 0 else 1
                self.neuron_class_freq[ineuron, icls] = (sim[idx] >= 0).sum() / den
                self.neuron_class_hits[ineuron, icls] = (sim[idx] >= 0).sum()
                den = float(idx.sum()) if idx.sum() != 0 else 1
                self.neuron_class_rate[ineuron, icls] = (sim[idx] >= 0).sum() / float(idx.sum())
            self.neuron_hits[ineuron] = (sim >= 0).sum()
            # Assign classes to the neurons
            imax = np.argmax(self.neuron_class_rate[ineuron])
            if self.neuron_class_rate[ineuron][imax] == 0.0:
                self.classes[ineuron] = -1
            else:
                self.classes[ineuron] = tgt_classes[imax]
        self.__print('Recoloring done in %i s'%(time.time()-t1))

    ######################################################################################
    def performance(self, data, target, fNormed = False):
        if not fNormed and self.trn_data_norm is not None:
            data = self.trn_data_norm.transform(data)

        sim, ineurons = self.winner(data, self.neurons, self.radius)
        self.__print('Performance:')
        # Loop over classes
        effs = np.zeros(np.unique(target).shape[0])
        for icls, cls in enumerate(np.unique(target)):
            idx = (sim >= 0) & (target == cls) & (self.classes[ineurons] == cls)
            effs[icls] = (100*idx.sum()/float((target == cls).sum()))
            self.__print('\tClass ' + str(cls) + ': %.2f%%'%(100*idx.sum()/float((target == cls).sum())))
            self.trn_info.metrics['PD_c%i'%cls] = effs[icls]
        # SP
        self.trn_info.metrics['SP'] = np.sqrt(np.power(np.prod(effs), 1.0/effs.shape[0])*np.mean(effs))
        self.__print('\tSP: %.2f%%'%(self.trn_info.metrics['SP']))
        self.trn_info.metrics['Unmapped'] = (100*(sim < 0).sum() / float(data.shape[0]))
        self.__print('\tUnmapped: %.2f%%'%(self.trn_info.metrics['Unmapped']))

    ######################################################################################
    def classify(self, data, fNormed = False):
        if not fNormed and self.trn_data_norm is not None:
            data = self.trn_data_norm.transform(data)

        sim, ineurons = self.winner(data, self.neurons, self.radius)
        classes = self.classes[ineurons]
        classes[sim < 0] = -1
        return classes, sim

    ######################################################################################
    def __performance_init(self, data, target, trnPar):
        self.trn_info.perf = 0.0
        self.trn_info.itrn = np.array(trnPar['itrn'],'i')
        self.trn_info.itst = np.array(trnPar['itst'],'i')
        self.trn_info.ival = np.array(trnPar['ival'],'i')
        self.trn_info.perf_type = ''
        self.trn_info.best_epoch = 0
        max_neuron = data.shape[0]
        for cls in np.unique(target):
            self.trn_info.metrics['phase1_eff_trn_c%i'%cls] = np.zeros(self.trn_max_it)*np.nan
            self.trn_info.metrics['phase1_eff_val_c%i'%cls] = np.zeros(self.trn_max_it)*np.nan
            self.trn_info.metrics['phase1_eff_tst_c%i'%cls] = np.zeros(self.trn_max_it)*np.nan
            self.trn_info.metrics['phase2_eff_trn_c%i'%cls] = np.zeros(max_neuron)*np.nan
            self.trn_info.metrics['phase2_eff_val_c%i'%cls] = np.zeros(max_neuron)*np.nan
            self.trn_info.metrics['phase2_eff_tst_c%i'%cls] = np.zeros(max_neuron)*np.nan
    ######################################################################################
    def __performance_resume_phase1(self, it, neurons, radius, data, target, itrn, ival, itst, cls):
        # Measure the classification efficiency for each data set
        # Train
        self.trn_info.metrics['phase1_eff_trn_c%i'%cls] = self.trn_info.metrics['phase1_eff_trn_c%i'%cls][:it]
        # Val
        self.trn_info.metrics['phase1_eff_val_c%i'%cls] = self.trn_info.metrics['phase1_eff_val_c%i'%cls][:it]
        # Tst
        self.trn_info.metrics['phase1_eff_tst_c%i'%cls] = self.trn_info.metrics['phase1_eff_tst_c%i'%cls][:it]
    ######################################################################################
    def __performance_monitor_phase1(self, it, neurons, radius, data, target, itrn, ival, itst, cls):
        # Measure the classification efficiency for each data set
        # Train
        sim, ineuron = self.winner(data[itrn], neurons, radius)
        self.trn_info.metrics['phase1_eff_trn_c%i'%cls][it] = (sim > 0).sum() / float(sim.shape[0])
        # Val
        sim, ineuron = self.winner(data[ival], neurons, radius)
        self.trn_info.metrics['phase1_eff_val_c%i'%cls][it] = (sim > 0).sum() / float(sim.shape[0])
        # Tst
        sim, ineuron = self.winner(data[itst], neurons, radius)
        self.trn_info.metrics['phase1_eff_tst_c%i'%cls][it] = (sim > 0).sum() / float(sim.shape[0])

    ######################################################################################
    """
        Find the optimal radius for the dataset given
    """
    def optimal_radius(self, X):
        rate = self.trn_max_radius_sample if self.trn_max_radius_sample != -1 else X.shape[0]
        n = int(X.shape[0] * rate)

        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        idx = idx[:n]

        D = self.__dist_function.pairwise(X[idx])

        D = D[np.triu_indices_from(D, 1)] # get triangular without diagonal
        (hcnt, hval) = np.histogram(D, 50)
        radius = hval[np.argmax(hcnt)] * self.trn_radius_factor
        if radius < self.trn_min_radius:
            radius = self.trn_min_radius
        else:
            self.trn_min_radius = 0.1 * radius
        return radius











