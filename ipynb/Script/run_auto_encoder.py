import sys

import numpy as np
import time
from sklearn.externals import joblib
from sklearn import model_selection
import itertools



np.set_printoptions(3)

sys.path.append('../')


from Sonar import StackedAutoEncoderCV as SAE



########################################### CONFIGURATION
known_classes = ['ClassD']
fsave = '../Models/single_' + '-'.join(known_classes) + '.jbl'
data_file = '/home/natmourajr/Public/Marinha/Data/DadosCiodaro/4classes/lofar_data_file_fft_1024_decimation_3_spectrum_left_400.jbl'
#hidden_layers = [
#    np.arange(80,120,10),
#    np.arange(30,70,10),
#    np.arange(20,30,2),
#]
hidden_layers = [
    [100],
    [50],
    [25],
]


optimizers = [['adam', 'adam','adam']]
nepochs = [500]
batch_size = [125]
ninit = [1]
seed = 10

########################################### LOAD DATA
dataset = joblib.load(data_file)
data  = dataset[0]
target= dataset[1]
labels= dataset[2]

########################################### SELECT ONE CLASS
id_classes = [k for k,v in labels.items() if v in known_classes]
idx = np.isin(target, id_classes) 

nov_data   = data[~idx]
nov_target = target[~idx]

data   = data[idx]
target = target[idx]

nclasses = np.unique(target).shape[0]

########################################### TRAINING INDEXES

# Test x Development
ntrn = 0.8
Xtrn, Xtst, Ytrn, Ytst = model_selection.train_test_split(data, target, test_size = 1.0-ntrn,
                                                          stratify=target, random_state = seed)
#Xtrn = Xtrn[:10]
#Ytrn = Ytrn[:10]

########################################## GRID-SEARCH
hiddens = [(Xtrn.shape[1],) + x + x[::-1][1:] + (Xtrn.shape[1],) for x in itertools.product(*hidden_layers)]


#raise Exception('STOP')

param_grid = {
    'hiddens': hiddens,
    'optimizers': optimizers,
    'nepochs': nepochs,
    'batch_size': batch_size,
    'ninit': ninit
}

########################################## TRAIN MODEL
print 'Running Grid Search on Stacked Auto Encoder'
for k,v in param_grid.items():
    print '\t#'+k+' conf.: ', len(v)

cvmodel = SAE.StackedAutoEncoderCV(param_grid, nfolds=5, njobs = 4, random_seed = seed)


cvmodel.fit(Xtrn, Ytrn, nclasses)
cvmodel.save(fsave)
print 'Final Score: %.2f +- %.2f'%(cvmodel.mean_score, cvmodel.std_score)

########################################## RUN NOVELTY
print '==== UNKNOWN CLASSES ===='
for cls in np.unique(nov_target):
    X = nov_data[nov_target == cls]
    score = cvmodel.score(X)
    print '\t'+labels[cls]+' score: %.2f'%(score)



    



    
    
# END OF FILE





