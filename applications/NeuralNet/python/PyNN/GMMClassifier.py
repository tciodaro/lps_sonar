

from . import TrnInfo
from sklearn import metrics
from sklearn import mixture
import numpy as np



class GMMClassifier(object):

    def __init__(self):
        self.trn_max_it = 100
        self.trn_min_cov = 0.001
        self.trn_max_gaussians = 10
        self.trn_max_init = 10
        self.trn_data_scaler = None
        self.trn_info = TrnInfo.TrnInfo()
        self.gmms = {} # per class
        self.gmm_classes = []
        self.classes = None

    def train(self, data, target, trnPar):
        # Train parameters
        parnames = ['trn_max_it', 'trn_min_cov','trn_max_gaussians','trn_max_init']
        for name in parnames:
            if trnPar.has_key(name):
                self.__dict__[name] = trnPar[name]
        # Indexes
        if not trnPar.has_key('itrn'):
            raise Exception('Cannot train without training indexes')
        itrn = trnPar['itrn']
        ival = trnPar['ival'] if trnPar.has_key('ival') else itrn
        itst = trnPar['itst'] if trnPar.has_key('itst') else ival
        # Scale data
        if self.trn_data_scaler is not None:
            self.trn_data_scaler.fit(data[itrn])
            data = self.trn_data_scaler.transform(data)
        # Loop over different classes
        self.classes = np.unique(target)
        for cls in self.classes:
            # Loop over number of Gaussians
            best_score = np.inf
            best_gmm = None
            for n in range(2, self.trn_max_gaussians):
                gmm = mixture.GMM(n_components=n, covariance_type='full', tol=0,
                                  n_iter=self.trn_max_it, n_init=self.trn_max_init,
                                  params='wmc', init_params='wmc', verbose=0,
                                  min_covar=self.trn_min_cov)
                gmm.fit(data[itrn])
                Y = gmm.predict(data[itrn])
                score = gmm.bic(data[itrn])
                #score = metrics.silhouette_score(data[itrn], Y)
                if score < best_score:
                    best_score = score
                    best_gmm = gmm
            # Save Classifier
            self.gmms[cls] = best_gmm
            self.gmm_classes += [cls]*best_gmm.n_components
        self.gmm_classes = np.array(self.gmm_classes)

        # Estimate classification performance
        Y = self.predict(data)
        self.trn_info.metrics['trn_sp'] = self.calculate_sp(data[itrn], target[itrn], Y[itrn])
        self.trn_info.metrics['tst_sp'] = self.calculate_sp(data[itst], target[itst], Y[itst])
        self.trn_info.metrics['val_sp'] = self.calculate_sp(data[ival], target[ival], Y[ival])
        self.trn_info.perf = (target == Y) / float(data.shape[0])
        self.trn_info.metrics['trn_eff'] = (target[itrn] == Y[itrn]) / float(len(itrn))
        self.trn_info.metrics['val_eff'] = (target[ival] == Y[ival]) / float(len(ival))
        self.trn_info.metrics['tst_eff'] = (target[itst] == Y[itst]) / float(len(itst))
        self.trn_info.itrn = np.array(itrn, copy=True)
        self.trn_info.itst = np.array(itst, copy=True)
        self.trn_info.ival = np.array(ival, copy=True)
        self.trn_info.perf_type = 'SP'



    def calculate_sp(self, data, target, labels):
        effs = np.zeros(self.classes.shape[0])
        for i, cls in enumerate(self.classes):
            idx = (target == cls)
            effs[i] = (idx & (labels == cls)).sum() / float(idx.sum())
        return np.sqrt(np.power(np.prod(effs), 1./effs.shape[0]) * np.mean(effs))





    def predict(self, data):
        Y = np.zeros((data.shape[0], self.gmm_classes.shape[0]))
        i = 0
        for icls, cls in enumerate(self.classes):
            gmm = self.gmms[cls]
            Y[:, i:i+gmm.n_components] = gmm.predict_proba(data)
            i = i + gmm.n_components
        return np.array([self.gmm_classes[np.argmax(row)] for row in Y])



    def probabilities(self, data):
        Y = np.zeros((data.shape[0], self.classes.shape[0]))
        for icls, cls in enumerate(self.classes):
            Y[:, icls] = np.max(self.gmms[cls].predict_proba(data), axis=0)
        return Y





# END OF FILE







