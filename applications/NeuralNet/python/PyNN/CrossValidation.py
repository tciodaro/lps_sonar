
from . import NeuralNet
import numpy as np
from sklearn.externals import joblib

class CVModel(object):
    def __init__(self, pars):
        if not pars.has_key('indexes'):
            raise Exception('Missing indexes in Cross Validation')
        self.indexes = pars['indexes']
        self.TrnPerc = pars['TrnPerc'] if pars.has_key('TrnPerc') else 0
        self.ValPerc = pars['ValPerc'] if pars.has_key('ValPerc') else 0
        self.results = []
        self.CVNSel = pars['CVNSel'] if pars.has_key('CVNSel') else 1
        self.verbose = pars['verbose'] if pars.has_key('verbose') else True
        self.savefile = pars['savefile'] if pars.has_key('savefile') else None
    ######################################################################################
    """
        Train the given model considering each cross validation selection.
        Returns a list containing the results for each cross validation evaluation.
    """
    def train(self, data, target, model, modelPar,trnPar):
        if not isinstance(trnPar, dict):
            print 'CrossValidation Error: training parameters is not a dict'
            return None
        if self.verbose:
            print 'Training with cross validation ', self.__class__.__name__
        cvidx =  self.get_indexes()
        for isel, cvsel in enumerate(cvidx.values()):
            if self.verbose: print 'Cross Validation Selection: ', isel
            trnPar['itrn'] = cvsel['ITrn']
            trnPar['ival'] = cvsel['IVal']
            trnPar['itst'] = cvsel['ITst']
            # Create model
            obj = model(modelPar)
            obj.train(data, target, trnPar)
            self.results.append(obj)
            if self.savefile is not None:
                print 'CV: Auto saving'
                joblib.dump(self, self.savefile, compress=9)
        return self.results
       
##########################################################################################
##########################################################################################
##########################################################################################
## CV Fold class
class CVFold(CVModel):
    """
        The equivalent of the leave-one-out method, but considering a 'fold' out.
    """
    def __init__(self, pars):
        if not pars.has_key('CVNFold'):
            raise Exception('Missing parameter in Cross Validation: CVNFold')
        super(CVFold, self).__init__(pars)
        self.CVNFold = pars['CVNFold']
        self.CVNSel = pars['CVNFold'] if not pars.has_key('CVNSel') else pars['CVNSel']
    ######################################################################################
    """
        Returns the train, test and validation indexes for each cross validation selection.
        If indexes is a dictionary, the indexes are generated per key and later concatenated
        in order to generate a single train, validation and test index set.
    """
    def get_indexes(self):
        indexes = None
        if not isinstance(self.indexes, dict):
            indexes = {'0': self.indexes}
        else:
            indexes = self.indexes
        # Loop over keys and generate indexes
        cvindexes = {}
        for isel in range(self.CVNSel):
            cvindexes[isel] = {'ITrn': [],
                               'IVal': [],
                               'ITst': []}
        for idx in indexes.values():
            cv = self._indexes(idx)
            for isel in range(self.CVNSel):
                cvindexes[isel]['ITrn'] += cv[isel][0].tolist()
                cvindexes[isel]['IVal'] += cv[isel][1].tolist()
                cvindexes[isel]['ITst'] += cv[isel][2].tolist()
        return cvindexes
    ######################################################################################
    """
        Divide the indexes into train, test and validation sets.
    """
    def _indexes(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array(idx, 'i')
        ## Create boxes
        box_cnt = int(idx.shape[0] / float(self.CVNFold))
        np.random.shuffle(idx)
        boxes = {}
        ibef = 0
        for ibox in range(self.CVNFold - 1):
            boxes[ibox] = idx[ibef:(ibox+1)*box_cnt].tolist()
            ibef = (ibox+1)*box_cnt
        boxes[self.CVNFold-1] = idx[ibef:].tolist()
        boxes = np.array(boxes.values())
        ## Permute the test box
        allboxes = np.arange(boxes.shape[0])
        cvidx = {}
        for isel in range(self.CVNSel):
            itrn = np.concatenate(boxes[np.setdiff1d(allboxes, [isel])])
            itst = boxes[isel]
            ival = boxes[isel]
            # Final setup
            cvidx[isel] = (np.array(itrn), np.array(ival), np.array(itst))
        return cvidx
##########################################################################################
##########################################################################################
##########################################################################################
## CV Multi Fold class
class CVMultiFold(CVFold):
    """
        Creates the CVFold object. The Cross Validation with folds divides the indexes
        into CVNFold groups. These groups are randomly selected as training, validation
        and test indexes according to the proportions in TrnPerc and ValPerc. This selection
        is done CVNSel times.

        A model is trained for each selection and results are available for later usage.
    """
    def __init__(self, pars):
        if not pars.has_key('CVNFold'):
            raise Exception('Missing parameter in Cross Validation: CVNFold')
        super(CVMultiFold, self).__init__(pars)
        self.CVNSel = pars['CVNSel'] if pars.has_key('CVNSel') else 1
    ######################################################################################
    """
        Divide the indexes into train, test and validation sets.
    """
    def _indexes(self, idx):
        if not isinstance(idx, np.ndarray):
            idx = np.array(idx, 'i')
        ## Create boxes
        box_cnt = int(idx.shape[0] / float(self.CVNFold))
        np.random.shuffle(idx)
        boxes = {}
        ibef = 0
        for ibox in range(self.CVNFold - 1):
            boxes[ibox] = idx[ibef:(ibox+1)*box_cnt].tolist()
            ibef = (ibox+1)*box_cnt
        boxes[self.CVNFold-1] = idx[ibef:].tolist()
        ## Make random selections
        iboxes = np.array(range(self.CVNFold), 'i')
        cvidx = {}
        for isel in range(self.CVNSel):
            np.random.shuffle(iboxes)
            ntrn = int(self.CVNFold * self.TrnPerc + 0.5)
            nval = int(self.CVNFold * self.ValPerc + 0.5)
            itrn = []
            for i in range(0, ntrn):
                itrn += boxes[iboxes[i]]
            ival = []
            for i in range(ntrn, ntrn+nval):
                ival += boxes[iboxes[i]]
            itst = []
            for i in range(ntrn+nval, self.CVNFold):
                itst += boxes[iboxes[i]]
            # Final setup
            cvidx[isel] = (np.array(itrn), np.array(ival), np.array(itst))
        return cvidx
