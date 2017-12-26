

"""
    Wrapper to load and write the nnet::NeuralNet.
    This class intends to load/write the parameters of
    the nnet::NeuralNet and detach its usage from its
    training.
"""
import os
import time
import sys

import neuralnet as nn

from sklearn.externals import joblib
import numpy as np
from . import ActFunctions as actfunc
from . import Initialization as init
from . import DataNorm as norm
from . import TrnInfo as PyTrnInfo

import multiprocessing as mp



class NeuralNet(object):
    def __init__(self, netpar = None):
        self.net_type = ''
        self._str_nodes = None
        self._str_funcs = None
        self.F = None # activation functions
        self.nlayers = 0
        self.nnodes = None
        self.W = None
        self.B = None
        self.useB = None
        self.useW = None
        self.frozenNode = None
        self.avgIn = None
        self.stdIn = None
        self.trn_info = None
        self.nnet = None # the C++ extended object to be trained.
        if netpar is not None:
            self.load(netpar)
    ###########################################################
    def createCNet(self):
        nnet = None
        if self.net_type.lower() == 'rprop':
            nnet = nn.RProp(self._str_nodes, self._str_funcs)
        elif self.net_type.lower() == 'backpropagation':
            nnet = nn.Backpropagation(self._str_nodes, self._str_funcs)
        nnet.initialize(1,-1)
        # Copy weights, bias...
        for ilay in range(len(self.W)-1):
            for i in range(self.W[ilay].shape[0]):
                nnet.setBias(ilay, i, self.B[ilay][i][0])
                nnet.setUseBias(ilay, i, bool(self.useB[ilay][i][0]))
                for j in range(self.W[ilay].shape[1]):
                    nnet.setWeight(ilay, i, j, self.W[ilay][i][j])
                    nnet.setUseWeights(ilay, i, j, bool(self.useW[ilay][i][j]))
                # Must freeze the node after setting it, otherwise it does not set it!
                # A Frozen node cannot have its values set
                nnet.setFrozen(ilay, i, bool(self.frozenNode[ilay][i][0]))
        return nnet
    ###########################################################
    """
        Print the network weights.
    """
    def print_weights(self):
        np.set_printoptions(formatter={'float': '{:+9.3f}'.format})
        for ilay in range(len(self.W)):
            print 'Layer ', ilay, ' ', '='*20
            for i in range(self.W[ilay].shape[0]):
                print self.B[ilay][i], '   ',
                print self.W[ilay][i]
        # reset
        np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8,
                            suppress=False, threshold=1000, formatter=None)
    ###########################################################
    """
        Copy the contents of the given net to this net.
    """
    def copy(self, nnet):
        self.net_type = nnet.net_type
        self.avgIn = np.array(nnet.avgIn)
        self.stdIn = np.array(nnet.stdIn)
        self.W = np.array(nnet.W)
        self.B = np.array(nnet.B)
        self.useW = np.array(nnet.useW)
        self.useB = np.array(nnet.useB)
        self._str_nodes = nnet._str_nodes
        self._str_funcs = nnet._str_funcs
        self.nlayers = nnet.nlayers
        self.nnodes = nnet.nnodes
        self.frozenNode = np.array(self.frozenNode)
        self._setActFunc(nnet.nlayers, nnet._str_funcs)
    ###########################################################
    """
        Feeds the neural network, simulating its response to the input given.
        Data must be in the format:
            X = [[Event 1], [Event 2], ..., [Event n]]
        Where each Event i is an array of input values.
        The output is in the format:
            Y = [[Out 1], [Out 2], ..., [Out n]]
        Where Out i is a vector with the network output.

        Always preprocess the events as if they should be removed the internal mean
        and rescaled by the interval standard deviation references.
    """
    def feedforward(self, X):
        # Transpose X
        Y = ((X - self.avgIn)/self.stdIn).transpose()
        Y[np.isnan(Y)] = 0
        for ilay in xrange(self.nlayers-1):
            Y = self.F[ilay]((self.W[ilay] * self.useW[ilay]).dot(Y) + self.B[ilay]*self.useB[ilay])
        return Y.transpose()
    ###########################################################
    """
        Writes the data do the file given. Apply the sklearn.externals.joblib.
    """
    def write(self, fname):
        if self._str_nodes is None:
            raise Exception('NeuralNet structure is empty. Nothing to save')
        joblib.dump(self, fname, compress=9)
    ###########################################################
    """
        Saves the nnet::neuralnet given do the file given. Apply the sklearn.externals.joblib.
    """
    @staticmethod
    def save(nnet, fname):
        pynet = NeuralNet()
        pynet.load(nnet)
        joblib.dump(pynet, fname, compress=9)
    ###########################################################
    """
        Read the NeuralNet considering the file given. Apply the sklearn.externals.joblib.
    """
    @staticmethod
    def read(self, fname):
        return joblib.load(fname)
    ###########################################################
    """
        Loads the nnet::NeuralNet given. If the input parameter
        is a string, consider it the filename from where to load
        the data.
    """
    def load(self, p):
        if isinstance(p, str):
            self._loadFromFile(p)
        elif isinstance(p, dict):
            self._buildNNet(p)
        else:
            self._loadFromNNet(p)
    ###########################################################
    """
        Load the nnet::NeuralNet from a file. The file should
        contain this class stored as a joblib object
    """
    def _loadFromFile(self, p):
        # file exists?
        if not os.path.exists(p):
            raise Exception('Could not open file: ' + p)
        nnet = joblib.load(p)
        self.copy(nnet)
    ###########################################################
    """
        Load the nnet::NeuralNet from the given reference.
    """
    def _loadFromNNet(self, p):
        self.net_type = p.__class__.__name__
        self._str_nodes = p.str_nodes
        self._str_funcs = p.str_funcs
        nLayers = p.getNLayers()
        self.nnodes = np.array([p.getNNodes(i) for i in range(nLayers)])
        if nLayers != len(self._str_nodes.split(':')) or nLayers-1 != len(self._str_funcs.split(':')):
            raise Exception('Number of layers and network strings do not match')
        self._setActFunc(nLayers, p.str_funcs)
        self.W = np.array([None] * (nLayers-1))
        self.B = np.array([None] * (nLayers-1))
        self.useB = np.array([None] * (nLayers-1))
        self.useW = np.array([None] * (nLayers-1))
        self.frozenNode = np.array([None] * (nLayers-1))
        for ilay in range(nLayers-1):
            # Initialize bias
            self.B[ilay]    = np.zeros((p.getNNodes(ilay+1), 1))
            self.useB[ilay] = np.ones((p.getNNodes(ilay+1), 1))
            self.frozenNode[ilay] = np.zeros((p.getNNodes(ilay+1), 1))
            # Initialize weights
            self.W[ilay] = np.zeros((p.getNNodes(ilay+1), p.getNNodes(ilay)))
            self.useW[ilay] = np.ones((p.getNNodes(ilay+1), p.getNNodes(ilay)))
            for i in range(p.getNNodes(ilay+1)):
                self.B[ilay][i,0] = p.getBias(ilay, i)
                self.useB[ilay][i,0] = p.getUseBias(ilay, i)
                self.frozenNode[ilay][i][0] = p.isFrozenNode(ilay, i)
                for j in range(p.getNNodes(ilay)):
                    self.W[ilay][i,j] = p.getWeight(ilay, i, j)
                    self.useW[ilay][i,j] = p.getUseWeight(ilay, i, j)
        self.avgIn = np.zeros(self.nnodes[0]) # standard values
        self.stdIn = np.ones(self.nnodes[0])
        self.nlayers = nLayers
    ###########################################################
    """
        Builds the neural network considering the parameters in the dictionary
    """
    def _buildNNet(self, p):
        if not p.has_key('builder') or not p.has_key('nodes') or not p.has_key('activ'):
            raise Exception('Cannot build network: missing parameters')
        self._str_nodes = p['nodes'].lower()
        self._str_funcs = p['activ'].lower()
        self.nnet = p['builder'](self._str_nodes, self._str_funcs)
        self.net_type = self.nnet.__class__.__name__
        self.nnet.initialize(1,-1)
        self._loadFromNNet(self.nnet)
    ###########################################################
    def _setActFunc(self, nLayers, str_funcs):
        # Activation Function
        if nLayers-1 != len(str_funcs.split(':')):
            raise Exception('Number of layers and function string does not match')
        self.F = [None]*(nLayers-1)
        for idx, val in enumerate(str_funcs.split(':')):
            if val.find('tanh') != -1:
                self.F[idx] = actfunc.tanh
            elif val.find('sigmoid') != -1:
                self.F[idx] = actfunc.sigmoid
            elif val.find('lin') != -1:
                self.F[idx] = actfunc.purelin
            elif val.find('winner') != -1:
                self.F[idx] = actfunc.winner
            else:
                raise Exception('Unknown given activation function: ' + val)
    ###########################################################
    """
        Trains the network with the data in X and the targets in T.
        X and T must be numpy.ndarrays with two dimensions, considering the events
        as first dimension and input variables as second dimension. Target data should
        follow the same convention. The train parameters are passed as a dictionary in
        p. The following parameters are available:

            'itrn': indexes for the training data (default: all events)
            'ival': indexes for the validation data (default is no validation)
            'itst': indexes for the test data (default is no test)
            'ninit': number of random weight initializations (default is 1)
            'fbatch': flags the training as batch, instead of online (default is True)
            'nepochs': number of training epochs (default is 100)
            'nshow': number of epochs to summarize the training (default is 10% of 'nepochs')
            'winit': method to use as weight initializer (default is initialization.initnw)
            'task': if network is trained for 'classification' or 'estimation' (default)
            'datanorm': which data normalization method to use (default is datanorm.mapstd)
            'performance': the performance type to consider (default is 'mse')
    """
    def train(self, X, T, p):
        if not self.nnet:
            raise Exception('NeuralNet: network was not created yet')
        # Check data and target
        if not isinstance(X, np.ndarray) or not isinstance(T, np.ndarray):
            raise Exception('NeuralNet: data or target are not numpy arrays')
        print 'NeuralNet: starting training (', self._str_nodes, ')... ',
        sys.stdout.flush()
        t0 = time.time()
        # Number of processes
        p = dict(p)
        nproc = p['nprocesses'] if p.has_key('nprocesses') else 4
        ninit = int(p['ninit']) if p.has_key('ninit') else 1
        p['builder'] = self.nnet.__class__.__name__
        p['nodes'] = self._str_nodes
        p['funcs'] = self._str_funcs
        pool = mp.Pool(processes = nproc)
        self.nnet = None # Not serializable
        processes = [pool.apply_async(proc_train, (X, T, p, i, self)) for i in range(ninit)]
        results = [pr.get() for pr in processes]
        perfs = [net.trn_info.perf for net in results]
        best_net = None
        if p['task'].lower() == 'estimation':
            best_net = results[np.argmin(perfs)]
        elif p['task'].lower() == 'classification':
            best_net = results[np.argmax(perfs)]
        # Transform data to C++ matrixes
        self.copy(best_net)
        self.trn_info = best_net.trn_info
        self.trn_info.itrn = np.array(p['itrn'])
        self.trn_info.ival = np.array(p['ival'])
        self.trn_info.itst = np.array(p['itst'])
        self.trn_info.metrics['init_perfs'] = np.array(perfs)
        pool.close()
        print ' done in %.3f s'%(time.time() - t0)
############################################################################
## TRAINING METHODS FOR MULTIPROCESSING

def proc_train(data, target, trnPar, iinit, pynnet):
    # Det default parameters
    newp = {}
    newp['itrn'] = nn.RowUI(trnPar['itrn'] if trnPar.has_key('itrn') else np.arange(len(data)))
    newp['ival'] = nn.RowUI(trnPar['ival'] if trnPar.has_key('ival') else [])
    newp['itst'] = nn.RowUI(trnPar['itst'] if trnPar.has_key('itst') else [])
    if not newp['ival'].size(): newp['ival'] = newp['itrn']
    if not newp['itst'].size(): newp['itst'] = newp['ival']
    newp['fbatch'] = bool(trnPar['fbatch']) if trnPar.has_key('fbatch') else True
    newp['nepochs'] = int(trnPar['nepochs']) if trnPar.has_key('nepochs') else 100
    newp['nshow'] = int(trnPar['nshow']) if trnPar.has_key('nshow') else int(nepochs*0.1)
    newp['winit_par'] = trnPar['winit_par'] if trnPar.has_key('winit_par') else {}
    newp['task'] = trnPar['task'].lower() if trnPar.has_key('task') else 'estimation'
    newp['min_epochs'] = trnPar['min_epochs'] if trnPar.has_key('min_epochs') else 0
    newp['datanorm'] = norm.get_func(trnPar['datanorm'] if trnPar.has_key('datanorm') else 'mapstd')
    newp['winit'] = init.get_func(trnPar['winit'] if trnPar.has_key('winit') else 'initnw')
    for k,v in trnPar.iteritems():
        if not newp.has_key(k):
            newp[k] = v
    Xtrn = np.array(data)[newp['itrn']]
    nnet = pynnet.createCNet()
    data = nn.MatrixF(data)
    target = nn.MatrixF(target)
    # IO manager
    iomgr = nn.IOMgr()
    if not iomgr.initialize(data, target):
        raise Exception('NeuralNet: error loading data to IO Manager')
    iomgr.set_trn(newp['itrn'])
    iomgr.set_tst(newp['itst'])
    iomgr.set_val(newp['ival'])
    # Train manager
    trnmgr = nn.Trainbp()
    trnmgr.net_task = newp['task'];
    trnmgr.set_iomgr(iomgr);
    trnmgr.fbatch = newp['fbatch']
    trnmgr.nepochs = newp['nepochs']
    trnmgr.min_epochs = newp['min_epochs']
    trnmgr.nshow   = newp['nshow']
    newp['winit'](nnet, Xtrn, newp['winit_par'], seed = (int(time.time()*1000)+iinit)%(2**32-1))
    trnmgr.set_net(nnet)
    trnmgr.initialize()
    trnmgr.train()
    nnet = NeuralNet(trnmgr.get_net())
    nnet.trn_info = PyTrnInfo.TrnInfo(trnmgr.get_trninfo())
    return nnet





# END OF FILE

