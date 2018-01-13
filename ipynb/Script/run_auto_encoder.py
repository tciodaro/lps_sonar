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
known_classes = ['ClassA']
data_file = '/home/natmourajr/Public/Marinha/Data/DadosCiodaro/4classes/lofar_data_file_fft_1024_decimation_3_spectrum_left_400.jbl'
hidden_layers = {
    'hidden_1': np.arange(80,120,10),
    'hidden_2': np.arange(40,80,10),
    'hidden_3': np.arange(20,30,2),
}
optimizers = ['adam', 'adam','adam']
nepochs = [10]
batch_size = [10]
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

########################################## GRID-SEARCH
hiddens = [(Xtrn.shape[0],) + x + (Xtrn.shape[0],) for x in itertools.product(*hidden_layers.values())]


param_grid = {
    'hiddens': hiddens,
    'optimizers': [['adam','adam','adam']],
    'nepochs': [500],
    'batch_size': [100],
    'ninit': [1]
}

########################################## TRAIN MODEL
print 'Running Grid Search on Stacked Auto Encoder'
for k,v in param_grid.items():
    print '\t#'+k+' conf.: ', len(v)

cvmodel = SAE.StackedAutoEncoderCV(param_grid, 1, 2, seed)

raise Exception("STOP")

cvmodel.fit(Xtrn, Ytrn, nclasses)
cvmodel.save('teste.jbl')
print 'Final Score: %.2f +- %.2f'%(cvmodel.mean_score, cvmodel.std_score)

########################################## RUN NOVELTY
print '==== UNKNOWN CLASSES ===='
for cls in nov_target.unique():
    X = nov_data[nov_target == cls]
    score = cvmodel.score(X)
    print '\t'+labels[k]+' score: %.2f'%(score)



    



    
    
# END OF FILE





