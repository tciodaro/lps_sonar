
import numpy as np
from . import PCD as PyPCD
from . import NeuralNet as PyNNet
import time

class PCD_IsoCooperative(PyPCD.PCD):
    """
        Isolated Cooperative PCD extraction.

        The PCDs are extraced combining the advantages of both the Constructive PCD,
        and the Independent PCD. The neural network used to extract the PCDs
        has tree layers: the first layer is responsible for compacting into the
        principal components, while the second and third layers are responsible
        behave like a MLP classifier. Both MLP and the PCD layer are trained considering
        the error backpropagation.

        Later, the PCD layer can be used separately from the MLP classifier. After
        extracting a given PCD, another neuron is added to the first layer and all
        other neurons are frozen. The PCDs are extracted removing the data projetion
        onto the extracted PCD and the PCD norm is normalized to 1.
    """
    def __init__(self, pars):
        super(PCD_IsoCooperative, self).__init__(pars)
    ###########################################################################
    """
        Create new network object and copy the needed values to accomodate a new component
    """
    def _add_new_pcd(self, nnet):
        npcd = nnet.W[0].shape[0]
        nhidden = nnet.W[1].shape[0]
        nout = nnet.W[2].shape[0]
        self.netPars['nodes'] = str(nnet.W[0].shape[1])+":"+ \
                        str(npcd+1)+ ":" + str(nhidden) +":"+str(nout)
        newnet = PyNNet.NeuralNet(self.netPars)
        newnet.nnet.initialize(1,-1)
        # Copy old weight/bias values
        for ilay in range(len(nnet.W)-1):
            for i in range(nnet.W[ilay].shape[0]):
                newnet.nnet.setBias(ilay, i, nnet.B[ilay][i][0])
                newnet.nnet.setUseBias(ilay, i, bool(nnet.useB[ilay][i][0]))
                newnet.B[ilay][:-1] = nnet.B[ilay]
                newnet.useB[ilay][:-1] = nnet.useB[ilay]
                for j in range(nnet.W[ilay].shape[1]):
                    newnet.nnet.setWeight(ilay, i, j, nnet.W[ilay][i][j])
                    newnet.nnet.setUseWeights(ilay, i,j, bool(nnet.useW[ilay][i][j]))
                    newnet.W[ilay][i][j] = nnet.W[ilay][i][j]
                    newnet.useW[ilay][i][j] = nnet.useW[ilay][i][j]
        # No Bias in the first layer
        newnet.B[0][:,0] = 0.0
        for i in range(newnet.B[0].shape[0]): newnet.nnet.setBias(0, i, 0.0)
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
        nhidden = 1 if not trnPar.has_key('nhidden') else trnPar['nhidden']
        nhidden = 1 if nhidden == -1 else nhidden
        # Create first net
        self.netPars['activ'] = 'lin:tanh:tanh'
        self.netPars['nodes'] = str(data.shape[1])+":1:"+\
                                str(nhidden)+":"+str(target.shape[1])
        trnPar['perftype'] = '' if not trnPar.has_key('perftype') else trnPar['perftype']

        print self.netPars['nodes']
        print self.netPars['activ']
        pynnet = PyNNet.NeuralNet(self.netPars)
        # Remove bias from first layer
        pynnet.useB[0][:,0] = 0
        pynnet.B[0][:,0] = 0.0
        pynnet.nnet.setUseBias(0, False)
        for i in range(pynnet.B[0].shape[0]): pynnet.nnet.setBias(0, i, 0.0)
        # LOOP OVER PCD EXTRACTION
        prevPerf = 0
        npcd = 1
        while True:
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
            if npcd >= self.MaxPCD:
                print 'PCD: maximum number of components extracted'
                pynnet = self._add_new_pcd(pynnet)
                break
            pynnet = self._add_new_pcd(pynnet)
            npcd = npcd + 1
            prevPerf = currPerf

        return pynnet
    """
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
    """


# End of file


