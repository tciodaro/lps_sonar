
import numpy as np
from . import NeuralNet as PyNNet
from . import CrossValidation as PyCV


class HiddenNeuronTrain(object):
    """
        Hidden neurons network training. This class implements a neural network
        training where the number of hidden neurons is constantly increased until it
        reaches a certain max value. Each trained network is created.

        The training supports cross validation techniques. Default to None, where the
        training indexes must be explicitly given in the training parameters.
    """
    def __init__(self,pars):
        self.nnets = {} # key: hidden neuron
        self.verbose = pars['Verbose'] if pars.has_key('Verbose') else True
        if not pars.has_key('HNeurons'):
            raise Exception('Error: missing HNeurons parameter')
        neurons = pars['HNeurons']
        self.netPars = pars['NNetParameters']
        for i in neurons:
            self.nnets[i] = None
    ######################################################################################
    """
        Train multiple networks for each hidden neuron.
    """
    def train(self, data, target, trnPar):
        if self.verbose:
            print 'Starting Hidden Neuron training'
        # Loop over hidden neurons
        nin = data.shape[1]
        nout = target.shape[1]
        for nh in self.nnets.keys():
            if self.verbose:
                print '=> Training for hidden neurons: ', nh
            nodes = str(nin)+':'+str(nh)+':'+str(nout)
            self.netPars['nodes'] = nodes
            net = PyNNet.NeuralNet(self.netPars)
            net.train(data, target, trnPar)
            self.nnets[nh] = net
            if self.verbose:
                print '==> Performance for net ', nodes, ' = %.3f'%net.trn_info.perf
        return self.nnets
