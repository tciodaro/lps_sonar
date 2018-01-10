from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend

import matplotlib.pyplot as plt

import numpy as np
import time
from sklearn import metrics


class StackedAutoEncoderCV(object):
    
    def __init__(self, grid_params, verbose = False):
        self.grid_params = grid_params
        self.__model = None
        self.__grid_obj = None
        self.verbose = verbose
        self.trn_perc = 0.2
           
    
    """
        Fit the grid search 
    """
    def fit(self, data, target=None):
        
        

        
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


