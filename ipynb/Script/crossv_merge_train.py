

from sklearn.externals import joblib
import os
import sys
import re
import PyNN.CrossValidation as PyCV

if len(sys.argv) != 4:
    print 'usage: ', sys.argv[0], ' <cvidx file> <novelty> <directory>'
    sys.exit(-1)
##################################################################
## GET ARGUMENTS
filecv = sys.argv[1]
novcls = sys.argv[2]
filedir= sys.argv[3]
##################################################################
## LOAD CV FILE WITH CONFIGURATION
cvconf = joblib.load(filecv)
##################################################################
## FINAL STRUCTURE
savedict = {}
savedict['Novelties'] = [novcls]
savedict['Classes'] = None
##################################################################
## CV STRUCTURE
cvPar = {}
cvPar['CVNFold'] = cvconf['CVNFold']
cvPar['CVNSel'] = cvconf['CVNSel']
cvPar['TrnPerc'] = cvconf['TrnPerc']
cvPar['ValPerc'] = cvconf['ValPerc']
cvPar['indexes'] = [] # dont need that

savedict['PCDModel']   = PyCV.CVModel(cvPar)
savedict['Classifier'] = PyCV.CVModel(cvPar)
savedict['ARTModel']   = PyCV.CVModel(cvPar)
savedict['ARTModel'].results = [None] * cvconf['CVNSel']
savedict['PCDModel'].results = [None] * cvconf['CVNSel']
savedict['Classifier'].results = [None] * cvconf['CVNSel']
savedict['cvPar'] = cvPar
##################################################################
## LOOP OVER NNET FILES
fsave = ''
for entry in os.listdir(filedir):
    # is the novelty class?
    if entry.find(novcls) == -1 or entry.find('_icv') == -1:
        continue
    # Get the CV index ('_icv%i')
    regex = re.search('_icv\d+', entry)
    icv = int(regex.group(0).replace('_icv','')) # shoud have only one group
    # Load
    print 'Loading file: ', entry
    obj = joblib.load(filedir + '/' + entry)
    # Fill train parameters, if any
    if not savedict.has_key('trnPar') and obj.has_key('trnPar'):
        savedict['Classes'] = obj['Classes']
        obj['trnPar']['itrn'] = []
        obj['trnPar']['ival'] = []
        obj['trnPar']['itst'] = []
        savedict['trnPar'] = obj['trnPar']
    # PCD
    if not savedict.has_key('pcdPar') and obj.has_key('pcdPar'):
        savedict['pcdPar'] = obj['pcdPar']
    # Hidden neurons training
    if not savedict.has_key('hiddenPar') and obj.has_key('hiddenPar'):
        savedict['hiddenPar'] = obj['hiddenPar']
    # Has a classifier key?
    if obj.has_key('Classifier'): savedict['Classifier'].results[icv] = obj['Classifier']
    if obj.has_key('PCDModel'): savedict['PCDModel'].results[icv] = obj['PCDModel']
    if obj.has_key('ARTModel'):
        savedict['ARTModel'].results[icv] = obj['ARTModel']
    # File name to save
    fsave = entry[:entry.find('_icv')]
## Save file
fsave = filedir + '/' + fsave + '.jbl'
joblib.dump(savedict, fsave, compress=9)

## END OF FILE


