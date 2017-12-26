
from sklearn.externals import joblib

class PCD(object):
    def __init__(self, pars):
        if not pars.has_key('MaxPCD') or not pars.has_key('NNetParameters'):
            raise Exception('Missing parameters to create PCD')
        self.MaxPCD = pars['MaxPCD']
        self.EvalDiff = pars['EvalDiff'] if pars.has_key('EvalDiff') else -1e6
        self.netPars = pars['NNetParameters'] if pars.has_key('NNetParameters') else None
        self.PCDNets = []
        self.PCDPerf = []
        self.PCDNum  = []
        self.verbose = pars['verbose'] if pars.has_key('verbose') else True
        self.savefile = pars['savefile'] if pars.has_key('savefile') else None
    ######################################################################################
    """
        Project the data in X into the npcd's.
        Returns the projected data.
    """
    def project(self, X, npcd):
        if len(self.PCDNets) < npcd:
            raise Exception('Invalid number of PCDs')
        W = self.PCDNets[npcd-1].W[0]
        return W.dot(X.transpose()).transpose()
    ######################################################################################
    def auto_save(self):
        if self.savefile is not None:
            nnPar = self.netPars
            self.netPars = None
            joblib.dump(self, self.savefile)
            self.netPars = nnPar


## END OF FILE

