"""
    Wrapper to load/write the Training information, from the NeuralNet library.
    It just loads the nnet::TrnInfo attributes to a python dictionary and read/write
    it to a file using the sklearn.externals.joblib module.
"""
import os

import neuralnet as nn
from sklearn.externals import joblib
import numpy as np

class TrnInfo(object):
    def __init__(self, trninfo = None):
        self.metrics = {}
        self.perf   = None
        self.data   = None
        self.target = None
        self.itrn = None
        self.itst = None
        self.ival = None
        self.perf_type = ''
        self.best_epoch = 0
        if trninfo is not None:
            self.load(trninfo)
    ###########################################################
    """
        To work as a dictionary
    """
    def __getitem__(self, key):
        return self.metrics[key]
        ###########################################################
    """
        To work as a dictionary
    """
    def __setitem__(self, key, value):
        self.metrics[key] = value
    ###########################################################
    """
        Copy the TrnInfo dictionary
    """
    def copy(self, trninfo):
        self.data    = trninfo.data
        self.metrics = trninfo.metrics
        self.best_epoch = trninfo.best_epoch
        self.itrn = trninfo.itrn
        self.ival = trninfo.ival
        self.itst = trninfo.itst
        self.target = trninfo.target
        self.perf_type = trninfo.perf_type
        self.perf = trninfo.performance()
    ###########################################################
    """
        Read the TrnInfo considering the file given. Apply the sklearn.externals.joblib.
    """
    @staticmethod
    def read(fname):
        return joblib.load(fname)
    ###########################################################
    """
        Write the TrnInfo considering the file given. Apply the sklearn.externals.joblib.
    """
    @staticmethod
    def save(trninfo, fname):
        if not len(trninfo.metrics):
            raise Exception('Empty TrainInfo')
        joblib.dump(trninfo, fname, compress=9)
    ###########################################################
    """
        Write this TrnInfo considering the file given. Apply the sklearn.externals.joblib.
    """
    def write(self, fname):
        if not len(self.metrics):
            raise Exception('Empty TrainInfo')
        joblib.dump(self, fname, compress=9)
    ###########################################################
    """
        Loads the nnet::TrnInfo given. If the input parameter
        is a string, consider it the filename from where to load
        the data.
    """
    def load(self, p):
        if isinstance(p, str):
            self._loadFromFile(p)
        else:
            self._loadFromTrnInfo(p)
    ###########################################################
    """
        Load the nnet::TrnInfo from a file. The file should
        contain this class stored as a joblib object
    """
    def _loadFromFile(self, p):
        # file exists?
        if not os.path.exists(p):
            raise Exception('Could not open file: ' + p)
        trninfo = joblib.load(p)
        self.copy(trninfo)
    ###########################################################
    """
        Load the nnet::TrnInfo from the given reference.
    """
    def _loadFromTrnInfo(self, p):
        for i in range(p.getNVar()):
            self.metrics[p.getVarName(i)] = np.array(p.getVarAddr(i), 'f')
        self.best_epoch = p.bst_epoch
        self.perf_type = p.perfType;
        self.perf = p.performance()

# END OF FILE

