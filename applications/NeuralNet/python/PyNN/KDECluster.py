
from sklearn.neighbors import KernelDensity
import numpy as np
from . import TrnInfo as PyInfo

class KDECluster(object):
    def __init__(self, par = None):
        self.density = None
        self.kde_nsamples = 1000
        self.trn_nsamples = 1000
        self.trn_info = PyInfo.TrnInfo()
        self.trn_data_norm = None
        self.knowledge_thr = 0 # minimal knowledge for known classes
        self.kde_bandwidth = 1.0
        self.kde_type = 'gaussian'
        self.trn_error = 0
        self.trn_error_perf = 0
        if par is not None:
            if isinstance(par, KDECluster):
                for key, val in par.__dict__.items():
                    setattr(self, key, val)
    ###########################################################################
    """
        Train the model to cluster the data.
    """
    def train(self, X, T, trnPar):
        print 'KDE Training'
        if not isinstance(trnPar, dict):
            print 'KDECluster: train parameter is not a dict'
            return
        if not isinstance(X, np.ndarray):
            X = np.array(X, 'f')
        if T is not None:
            if not isinstance(T, np.ndarray):
                T = np.array(T, 'f')
            if X.shape[0] != T.shape[0]:
                raise Exception('Training data and target have different sizes')
        self._set_parameters(X, trnPar)
        ## Initialize
        X,T = self._initialize(X, T)
        # Train density
        print 'Estimating Density'
        self.density = KernelDensity(kernel=self.kde_type, bandwidth=self.kde_bandwidth)
        itrn = self.trn_info.itrn
        idx = np.arange(itrn.shape[0])
        np.random.shuffle(idx)
        Xtrn = X[itrn]
        nsamples = self.trn_nsamples if self.trn_nsamples < itrn.shape[0] else itrn.shape[0]
        self.density.fit(Xtrn[idx[:nsamples]])
        # Evaluate domain
        randomsamples = self.density.sample(self.kde_nsamples)
        Zsim  =  np.exp(self.density.score_samples(randomsamples))
        Zsim.sort()
        zthrs = np.linspace(Zsim.min(), Zsim.max(), 10000)
        zsum = np.array([[(Zsim >= th).sum() / float(Zsim.size), th] for th in zthrs])
        iarg = np.argmax(zsum[zsum[:,0] >= self.knowledge_thr, 1])
        self.kde_thrs = Zsim[np.nonzero(Zsim >= zsum[iarg,1])][0]
        # Calculate error
        Zreal =  np.exp(self.density.score_samples(X))
        Zreal.sort()
        zthrs = np.linspace(Zreal.min(), Zreal.max(), 10000)
        zsum = np.array([[(Zreal >= th).sum() / float(Zreal.size), th] for th in zthrs])
        iarg = np.argmax(zsum[zsum[:,0] >= self.knowledge_thr, 1])
        real_thr = Zreal[np.nonzero(Zreal >= zsum[iarg,1])][0]
        self.trn_error = (self.kde_thrs - real_thr) / (real_thr if real_thr != 0 else 1)
        idx = self.trn_info.itst if len(self.trn_info.itst) else np.arange(X.shape[0])
        Z = np.exp(self.density.score_samples(X[idx]))
        self.trn_info.perf = np.sum(Z[idx] > self.kde_thrs) / float(X.shape[0])
        real_perf = np.sum(Z[idx] > real_thr) / float(Z.shape[0])
        self.trn_error_perf = np.abs(self.trn_info.perf - real_perf)
        print 'KDECluster: performance = %.4f'%self.trn_info.perf
        print 'KDECluster: threshold error = %.4f'%(self.trn_error)
        print 'KDECluster: performance error = %.4f'%self.trn_error_perf
    ######################################################################################
    """
        Classify the input
    """
    def classify(self, X):
        if self.trn_data_norm is not None:
            X = self.trn_data_norm.transform(X)
        Z = np.exp(self.density.score_samples(X))
        return Z > self.kde_thrs
    ######################################################################################
    """
        Feedforward
    """
    def feedforward(self, X):
        if self.trn_data_norm is not None:
            X = self.trn_data_norm.transform(X)
        return np.exp(self.density.score_samples(X))
    ######################################################################################
    """
        Set the training parameters
    """
    def _initialize(self, X, T):
        if self.trn_data_norm is not None:
            self.trn_data_norm.fit(X[self.trn_info.itrn])
            X = self.trn_data_norm.transform(X)
        return (X,T)
    ######################################################################################
    """
        Set the training parameters
    """
    def _set_parameters(self, data, trnPar):
        if trnPar is not None:
            if isinstance(trnPar, dict):
                for key, val in trnPar.items():
                    if key in self.__dict__.keys():
                        setattr(self, key, val)
        self.trn_info.itrn = trnPar['itrn'] if trnPar.has_key('itrn') else np.arange(data.shape[0])
        if trnPar.has_key('ival'):
            self.trn_info.ival = trnPar['ival'] if len(trnPar['ival']) else self.trn_info.itrn
        else:
            self.trn_info.ival = self.trn_info.itrn
        if trnPar.has_key('itst'):
            self.trn_info.itst = trnPar['itst'] if len(trnPar['itst']) else self.trn_info.ival
        else:
            self.trn_info.itst = np.arange(data.shape[0])
        self.trn_info.itrn = np.array(self.trn_info.itrn)
        self.trn_info.ival = np.array(self.trn_info.ival)
        self.trn_info.itst = np.array(self.trn_info.itst)  
            

## END OF FILE




