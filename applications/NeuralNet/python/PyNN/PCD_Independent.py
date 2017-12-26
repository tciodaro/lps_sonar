

from . import PCD as PyPCD
from . import NeuralNet as PyNNet
from . import ActFunctions as PyAct
import time
import numpy as np

class PCD_Independent(PyPCD.PCD):
    """
        Independent PCD Extraction.

        The components are extracted considering a Input:1:Output network architecture.
        The first components is extracted in order to maximize the detection of the given
        classes. After extraction, the second component is extracted after removing the
        data projection onto the previous extracted component. This process is done until
        the last component is extracted. Each component should be normalized to unity.

        Different from the Constructive method, this approach needs a final classifier
        to combine the data projetion onto each component and to maximize classification.

    """
    def __init__(self, pars):
        super(PCD_Independent, self).__init__(pars)
    ###########################################################################
    """
        Train the PCD model.
    """
    def train(self, data, target, trnPar):
        # Create net
        self.netPars['activ'] = 'tanh:tanh'
        self.netPars['nodes'] = str(data.shape[1])+":1:"+str(target.shape[1])
        prevPerf = 0
        # Loop over PCD Extraction
        W = np.zeros((self.MaxPCD, data.shape[1])) # PCD Transformation
        trnPar['winit_par'] = {'W': [None, None]}
        trnPar['perftype'] = '' if not trnPar.has_key('perftype') else trnPar['perftype']
        trndata = np.array(data)
        for npcd in range(1, self.MaxPCD+1):
            if self.verbose:
                print 'PCD: extracting pcd ', npcd
            # Train
            t0 = time.time()
            pynnet = PyNNet.NeuralNet(self.netPars)
            pynnet.useB[0][0] = 0
            pynnet.nnet.setUseBias(0,0,False)
            pynnet.nnet.initialize(1,-1)
            if npcd != 1:
                trnPar['winit_par']['W'] = [W[:npcd-1], None]
            pynnet.train(trndata, target, trnPar)
            if self.verbose:
                print 'PCD: performance with ', npcd, ': %.4f'%(pynnet.trn_info.perf),
                print ' took %.2f s'%(time.time()-t0)
            # Normalize PCD extracted and remove projection from input data
            W[npcd-1] = pynnet.W[0] / (np.linalg.norm(pynnet.W[0],2))
            trndata = trndata - np.outer(np.array([W[npcd-1]]).T, trndata.dot(W[npcd-1])).T
            

                
            print 'PROJECTION ONTO PCD'
            for i in range(npcd):
                print '\tFor ',i, ' pcd: ', np.mean(np.outer(np.array([W[i]]).T, trndata.dot(W[i])).T)
            
            # Create network to this PCD Transformation
            self._storePCDNet(W, npcd, pynnet)
            self.auto_save()
        # Finished
        self.netPars = None
        print 'PCD training finished.'
    ###########################################################################
    """
        Create a network to store the PCD transformation
    """
    def _storePCDNet(self, W, npcd,nnet):
        pynnet = PyNNet.NeuralNet()
        pynnet._str_nodes = str(W.shape[1])+':'+str(npcd)
        pynnet._str_funcs = 'purelin'
        pynnet.F = [PyAct.purelin]
        pynnet.nlayers = 2
        pynnet.nnodes = npcd
        pynnet.W = np.array([W[:npcd]])
        pynnet.B = np.array([np.zeros((npcd,1))])
        pynnet.useB = np.array([np.ones((npcd,1))])
        pynnet.useW = np.array([np.ones(W[:npcd].shape)])
        pynnet.avgIn = np.zeros(W.shape[1])
        pynnet.stdIn = np.ones(W.shape[1])
        self.PCDNets.append(pynnet)
        self.PCDNum.append(npcd)
        self.PCDPerf.append(nnet.trn_info.perf)



# End of file


