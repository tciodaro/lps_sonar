# From mat to py
import os
from scipy.io import loadmat
from sklearn.externals import joblib
import numpy as np


# Data loading
sonardir = os.getenv('SONARHOME')
fname   = sonardir + '/data/lofar_sonar_data_2048fft_3dec.mat'
fsave = sonardir + '/data/lofar_data_2048fft_3dec_fromMat.jbl'
mdata = loadmat(fname);
# Convert matlab data to python data structure
newdata = {}
ships = ['ClasseA', 'ClasseB','ClasseC','ClasseD']
for iship, ship in enumerate(ships):
    newdata[ship] = {}
    # Over ship
    # This might not be the run number, but rather the run position in the array
    # Must be corrected outside if there is a run missing in the data.
    for irun in range(mdata['raw_data_lofar'][0,0][iship].shape[1]):
        run = {}
        run['Fs'] = mdata['Fs'][0,0]
        run['DecRate'] = mdata['decimation_rate'][0,0]
        run['Overlap'] = mdata['num_overlap'][0,0]
        run['NFFT'] = mdata['n_pts_fft'][0,0]
        run['DecFIROrder'] = 10
        [F, T] = mdata['raw_data_lofar'][0,0][iship][0,irun].shape
        run['Freqs'] = np.linspace(0, run['Fs']/2.0, F)
        run['Timew'] = np.linspace(0, T * 1./run['Fs'], T)
        run['Signal'] = mdata['raw_data_lofar'][0,0][iship][0,irun]
        newdata[ship][irun] = run
# Save new file
joblib.dump(newdata, fsave, compress=9)



# end of file

