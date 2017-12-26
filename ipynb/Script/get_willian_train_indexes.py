
import os
from sklearn.externals import joblib


fname = os.getenv('SONARHOME') + '/data/lofar_data_2048fft_3dec_fromMat.jbl'
fsave = os.getenv('SONARHOME') + '/data/willian_thesis_indexes_2048fft.jbl'
classes = ['ClasseA', 'ClasseB','ClasseC','ClasseD']

##########################################################################################
## RUNS USED AS TRAIN/VALIDATION, ACCORDING TO WILLIAN'S PhD Thesis
## TRAIN: ODD TIME WINDOWS, VALIDATION: EVEN TIME WINDOWS
##########################################################################################
# Runs
trn_runs = {'ClasseA':[0, 1, 4],
            'ClasseB':[0, 4, 5, 6],
            'ClasseC':[1, 3, 5, 7, 9],
            'ClasseD':[3, 6, 8, 9]}
"""
A: N10, N11, N14
B: N20, N24, N25, N26
C: N31, N33, N35, N37, N39
D: N43, N46, N48, N49
"""


# Structure to be saved
trn = {}
val = {}
tst = {}

# Load data
data = joblib.load(fname)

# Loop over classes
for cls in classes:
    trn[cls] = {}
    val[cls] = {}
    tst[cls] = {}
    tottrn = 0
    tottst = 0
    for irun, rundata in data[cls].iteritems():
        idx = range(rundata['Signal'].shape[1])
        if not irun in trn_runs[cls]: # then, it is test
            trn[cls][irun] = []
            val[cls][irun] = []
            tst[cls][irun] = idx
            tottst = tottst + len(idx)
            continue
        # Train and validation:
        trn[cls][irun] = idx[::2]
        val[cls][irun] = idx[1::2]
        tst[cls][irun] = []
        tottrn = tottrn + len(idx)
    print cls
    print '\tTotal Runs: ', len(data[cls])
    print '\tTrain Perc: ', float(tottrn) / (tottrn+tottst)
    print '\tntrain: ', tottrn
    print '\tntest: ', tottst

indexes = {'Train': trn, 'Validation': val, 'Test': tst}
joblib.dump(indexes, fsave, compress=9)



## END OF FILE


