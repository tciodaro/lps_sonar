from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend

import matplotlib.pyplot as plt

import numpy as np
import time
from sklearn import metrics


class StackedAutoEncoder(object):
    
    def __init__(self, hiddens = [], optimizers = [], nepochs = 500, batch_size=100, ninit = 10, verbose=False):
        self.hiddens = hiddens
        self.optimizers = optimizers
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.ninit=ninit
            
        self.__model = None
        self.__encoder = None
        self.__trn_info = None
        self.__internal_nets = None
        self.verbose = verbose
       
    
    
    """
        Fit the auto encoder to the data given
    """
    def fit(self, data, target=None):
        t0 = time.time()
        if self.hiddens is None or len(self.hiddens) == 0:
            raise Exception('StackedAutoEncoder: hidden layers parameter not set')
        
        # Over training layers
        npairs = int((len(self.hiddens)-1)/2.0)
        self.__trn_info = {}
        nnet   = {}
        if self.verbose:
            print 'Training %i layer pairs'%npairs
        X = data
        for ilayer in range(1, npairs+1):
            if self.verbose:
                print "\tLayer pair %i"%(ilayer)
                print "\t\tStructure = %i:%i:%i"%(self.hiddens[ilayer-1],
                                                  self.hiddens[ilayer],
                                                  self.hiddens[len(self.hiddens) - ilayer])
                print "\t\tActivations = tanh:%s"%('linear' if ilayer == 1 else 'tanh') # only first iteration's output is linear
            ###### Training
            # Different Initializations
            self.__trn_info[ilayer] = None
            best_perf = 1e9
            nnet[ilayer] = None
            for iinit in range(self.ninit):
                model = models.Sequential()
                # Create network structure
                model.add(layers.Dense(self.hiddens[ilayer],
                                       activation = 'tanh',
                                       input_shape = [self.hiddens[ilayer-1]],
                                       kernel_initializer = 'uniform'))
                model.add(layers.Dense(self.hiddens[len(self.hiddens) - ilayer],
                                       activation = 'linear' if ilayer == 1 else 'tanh',
                                       input_shape = [self.hiddens[ilayer]],
                                       kernel_initializer = 'uniform'))  
                # Training
                opt = None
                if self.optimizers[ilayer-1] == 'adam':
                    opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
                elif self.optimizers[ilayer-1] == 'sgd':
                    opt = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
                else:
                    raise Exception('StackedAutoEncoder: unknown optimizer for pair %i: %s'%(ilayer-1,
                                                                                             self.optimizers[ilayer-1]))
                model.compile(loss='mean_squared_error', optimizer=opt)
                # Should be done for each initialization
                init_trn_desc = model.fit(X, X, 
                                          epochs = self.nepochs, 
                                          batch_size = self.batch_size,
                                          verbose = self.verbose,
                                          shuffle=False)
                # Get best
                if init_trn_desc.history['loss'][-1] < best_perf:
                    self.__trn_info[ilayer] = init_trn_desc
                    best_perf = init_trn_desc.history['loss'][-1]
                    nnet[ilayer] = model
            if self.verbose:
                print '\t\tTraining Error: %.3e'%(self.__trn_info[ilayer].history['loss'][-1])
            # Update input as the output of the hidden layer
            hidden_layer = backend.function([nnet[ilayer].layers[0].input],[nnet[ilayer].layers[0].output])
            X = hidden_layer([X])[0]
        # Put together final model
        self.__model = models.Sequential()
        self.__encoder = models.Sequential()
        for ilayer in range(1, npairs+1):
            # Encoder part
            self.__model.add(nnet[ilayer].layers[0])
            self.__encoder.add(nnet[ilayer].layers[0])
        for ilayer in range(npairs, 0, -1):
            # Decoder part
            self.__model.add(nnet[ilayer].layers[1])
        self.__internal_nets = nnet
        Y = self.__model.predict(data)
        score = metrics.mean_squared_error(data, Y)
        if self.verbose:
            print 'Final Reconstruction Error (training)  : %.3e'%(score)
            print 'Training took %i s'%(time.time() - t0)

        
    def encode(self, data):
        return self.__encoder.predict(data)
    
    def predict(self, data):
        return self.__model.predict(data)
    
    def score(self, data, target = None):
        Y = self.predict(data)
        return -metrics.mean_squared_error(Y, data)
    
    def get_params(self, deep=True):        
        return {'hiddens': self.hiddens,
                'optimizers': self.optimizers,
                'nepochs': self.nepochs,
                'batch_size': self.batch_size,
                'ninit': self.ninit}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_encoder(self):
        return self.__encoder
    
    def get_auto_encoder(self):
        return self.__model
    
    def plot_training_curves(self):
        for ilayer in self.__trn_info.keys():
            plt.figure()
            metric = 'loss'
            plt.plot(self.__trn_info[ilayer].epoch, self.__trn_info[ilayer].history[metric], '-b', lw = 3, label='train')
            plt.yscale('log')
            plt.legend(loc='best')
            plt.grid()
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Net %i:%i:%i - error (training) = %.3e'%(self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_output_shape_at(0)[1],
                                                                self.__model.layers[ilayer-1].get_input_shape_at(0)[1],
                                                                self.__trn_info[ilayer].history[metric][-1]))



# end of file


