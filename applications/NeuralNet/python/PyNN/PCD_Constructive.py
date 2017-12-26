
import numpy as np
from . import PCD as PyPCD
from . import NeuralNet as PyNNet
import time

class PCD_Constructive(PyPCD.PCD):
    """
        Constructive PCD

        A neural network is trained and new neurons are added at the hidden
        layer to estimate new components. The network starts with 1 hidden
        neuron, to estimate the first principal component. After training
        to maximize detection, a new hidden neuron is added and the weights
        of the previous neurons are frozen. The output layer and the weights
        of the next component are trained normally. This process continues
        until the network performance is considered irrelevant.

        The weights connecting the input to the hidden layer form the basis
        for the principal discrimination components. The data can be projected
        to the PCD basis with the matrix formed by these weights. The hidden
        neuron biases are not used.
    """
    def __init__(self, pars):
        super(PCD_Constructive, self).__init__(pars)
    ###########################################################################
    """
        Create new network object and copy the needed values to accomodate a new component
    """
    def _add_new_pcd(self, nnet):
        npcd = nnet.W[0].shape[0]
        nout = nnet.W[-1].shape[0]
        self.netPars['nodes'] = str(nnet.W[0].shape[1])+":"+str(npcd+1)+":"+str(nout)
        newnet = PyNNet.NeuralNet(self.netPars)
        newnet.nnet.initialize(1,-1)
        # Copy old weight/bias values
        for ilay in range(len(nnet.W)-1):
            for i in range(nnet.W[ilay].shape[0]):
                newnet.nnet.setBias(ilay, i, nnet.B[ilay][i][0])
                newnet.B[ilay][:-1] = nnet.B[ilay]
                for j in range(nnet.W[ilay].shape[1]):
                    newnet.nnet.setWeight(ilay, i, j, nnet.W[ilay][i][j])
                    newnet.W[ilay][i][j] = nnet.W[ilay][i][j]
        # Freeze nodes
        nodes = np.ones((npcd+1, 1))
        nodes[-1] = 0
        newnet.frozenNode = np.array(nnet.frozenNode)
        newnet.frozenNode[0] = nodes
        for i in range(npcd):
            newnet.nnet.setFrozen(0, i, bool(nodes[i]))

        return newnet
    ###########################################################################
    """
        Train the PCD model.

        Returns a n array with the results of
    """
    def train(self, data, target, trnPar):
        npcd = 1
        # Create first net
        self.netPars['activ'] = 'tanh:tanh'
        self.netPars['nodes'] = str(data.shape[1])+":"+str(npcd)+":"+str(target.shape[1])
        trnPar['perftype'] = '' if not trnPar.has_key('perftype') else trnPar['perftype']
        pynnet = PyNNet.NeuralNet(self.netPars)
        prevPerf = 0
        while True:
            # Check maximum number of PCD
            if npcd > self.MaxPCD:
                print 'PCD: maximum number of PCDs reached'
                break
            if self.verbose:
                print 'PCD: extracting pcd ', npcd
            # Train network
            t0 = time.time()
            pynnet.train(data, target, trnPar)
            currPerf = pynnet.trn_info.perf
            if self.verbose:
                print 'PCD: performance with ', npcd, ': %.4f'%(currPerf),
                print ' took %.2f s'%(time.time()-t0)
            # Should I add a new component
            # Check performance change
            if currPerf - prevPerf < self.EvalDiff:
                print 'PCD: irrelevant performance change: %.3f'%(currPerf - prevPerf)
                break
            # Store results
            self.PCDNets.append(pynnet)
            self.PCDNum.append(npcd)
            self.PCDPerf.append(currPerf)
            self.auto_save()
            pynnet = self._add_new_pcd(pynnet)
            npcd = npcd + 1
            prevPerf = currPerf
        # Finished
        self.netPars = None
        print 'PCD training finished.'



# End of file


