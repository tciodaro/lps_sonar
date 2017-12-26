import numpy as np
import neuralnet as nn
import sys


def get_func(name):
    if name is None:
        return no_init
    if name.lower() == 'inituni_ortho': return inituni_ortho
    if name.lower() == 'inituni': return inituni
    if name.lower() == 'initnw': return initnw
    return None


    
def gram_schmidt(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


##########################################################################################
def no_init(nnet, X, par = None, seed = -1):
    return
##########################################################################################
def inituni_ortho(nnet, X, par = None, seed = -1):
    """
    Initialize the weighes and bias considering an uniform distribution.
    X = [number of events x number of variables]

    Make the random vectors orthogonal to the ones received in par.
    """
    if seed != -1:
        np.random.seed(seed)
    nlayers = nnet.getNLayers()
    if par is None:
        orthoW = [None,None]
    else:
        orthoW = par['W'] if par.has_key('W') else None
    for ilay in range(nlayers-1):
        B = np.random.uniform(-1, 1, nnet.getNNodes(ilay+1))
        W = np.random.uniform(-1, 1, (nnet.getNNodes(ilay+1), nnet.getNNodes(ilay)))
        if orthoW[ilay] is not None:
            aux = np.vstack((orthoW[ilay], W))
            W = gram_schmidt(aux)[-1:]
        for i in range(nnet.getNNodes(ilay+1)):
            nnet.setBias(ilay, i, B[i]) # it sets only if not frozen
            for j in range(nnet.getNNodes(ilay)):
                nnet.setWeight(ilay, i, j, W[i,j]) # it sets only if not frozen

    #print 'IN INIT'
    #print '\tFor ',i, ' pcd: ', np.mean(np.outer(W.T, X.dot(W)).T)
                
##########################################################################################
def inituni(nnet, X, par = None, seed = -1):
    """
    Initialize the weighes and bias considering an uniform distribution.
    X = [number of events x number of variables]
    """
    if seed != -1:
        np.random.seed(seed)
    nlayers = nnet.getNLayers()
    for ilay in range(nlayers-1):
        B = np.random.uniform(-1, 1, nnet.getNNodes(ilay+1))
        W = np.random.uniform(-1, 1, (nnet.getNNodes(ilay+1), nnet.getNNodes(ilay)))
        for i in range(nnet.getNNodes(ilay+1)):
            nnet.setBias(ilay, i, B[i]) # it sets only if not frozen
            for j in range(nnet.getNNodes(ilay)):
                nnet.setWeight(ilay, i, j, W[i,j]) # it sets only if not frozen

##########################################################################################
def initnw(nnet, X, par = None, seed = -1):
    """
    Initialize weights and bias with Nguyen-Widrow rule.
    X = [number of events x number of variables]
    """
    if seed != -1:
        np.random.seed(seed)    
    nlayers = 3
    if nnet.getNLayers() != nlayers:
        print 'Nguyen-Widrow initialization set for 3 layers nets only'
        return

    str_funcs = nnet.str_funcs
    # First layers to hidden
    funcname = str_funcs.split(':')[0]
    active = np.array([-1,1])
    if funcname.lower() == 'purelin': active = np.array([-10,10])
    if funcname.lower() == 'sigmoid': active = np.array([0,1])
    inp = np.array([np.min(X,axis=0), np.max(X,axis=0)]).transpose()
    k = 0.7 * nnet.getNNodes(1)**(1./nnet.getNNodes(0))
    scale = (active[1] - active[0]) / (inp[:, 1] - inp[:, 0])
    w = k * np.random.uniform(-1, 1, (nnet.getNNodes(1), nnet.getNNodes(0)))
    b = k * np.linspace(-1, 1, nnet.getNNodes(1))
    # Scale
    x = 0.5 * (active[1] - active[0])
    y = 0.5 * (active[1] + active[0])
    w = w * x
    b = b * x + y
    # set
    for i in range(nnet.getNNodes(1)):
        nnet.setBias(0, i, b[i]) # it sets only if not frozen
        for j in range(nnet.getNNodes(0)):
            nnet.setWeight(0, i, j, w[i,j]) # it sets only if not frozen
    ## Output Layer
    funcname = str_funcs.split(':')[1]
    active = np.array([-1,1])
    if funcname.lower() == 'purelin': active = np.array([-10,10])
    if funcname.lower() == 'sigmoid': active = np.array([0,1])
    inp = np.tile(active, (nnet.getNNodes(1),1))
    k = 0.7 * nnet.getNNodes(2)**(1./nnet.getNNodes(1))
    scale = (active[1] - active[0]) / (inp[:, 1] - inp[:, 0])
    w = k * np.random.uniform(-1, 1, (nnet.getNNodes(2), nnet.getNNodes(1)))
    b = k * np.linspace(-1, 1, nnet.getNNodes(2))
    # Scale
    x = 0.5 * (active[1] - active[0])
    y = 0.5 * (active[1] + active[0])
    w = w * x
    b = b * x + y
    # set
    for i in range(nnet.getNNodes(2)):
        nnet.setBias(1, i, b[i])
        for j in range(nnet.getNNodes(1)):
            nnet.setWeight(1, i, j, w[i,j])
##########################################################################################


