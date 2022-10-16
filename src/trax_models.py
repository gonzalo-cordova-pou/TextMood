import trax

# set random seeds to make this notebook easier to replicate
# trax.supervised.trainer_lib.init_random_number_generators(31)

import trax.fastmath.numpy as np
from trax import layers as tl
from trax import fastmath


class Layer(object):
    """ Base class for layers.
    """
    def __init__(self):
        # set weights to None
        self.weights = None

    def forward(self, x):
        '''The forward propagation should be implemented
        by subclasses of this Layer class'''
        raise NotImplementedError

    def init_weights_and_state(self, input_signature, random_key):
        '''This function initializes the weights
        based on the input signature and random key,
        should be implemented by subclasses of this Layer class'''
        pass

    def init(self, input_signature, random_key):
        '''This initializes and returns the weights, do not override.'''
        self.init_weights_and_state(input_signature, random_key)
        return self.weights
    # __call__ allows an object of this class
    # to be called like it's a function.
    def __call__(self, x):
        # When this layer object is called,
        # it calls its forward propagation function
        return self.forward(x)


class Relu(Layer):
    """Relu activation function implementation"""
    def forward(self, x):
        '''
        Input:
            - x (a numpy array): the input
        Output:
            - activation (numpy array): all positive or 0 version of x'''
        activation = np.maximum(x,0)
        return activation


class Dense(Layer):
    """
    A dense (fully-connected) layer.
    """

    def __init__(self, n_units, init_stdev=0.1):
        '''__init__ is implemented for you'''
        # Set the number of units in this layer
        self._n_units = n_units
        self._init_stdev = init_stdev

    def forward(self, x):

        # Matrix multiply x and the weight matrix
        dense = np.dot(x, self.weights)    
        return dense

    # init_weights
    def init_weights_and_state(self, input_signature, random_key):    
        # The input_signature has a .shape attribute that gives the shape as a tuple
        input_shape = input_signature.shape

        # Generate the weight matrix from a normal distribution,
        # and standard deviation of 'stdev'  
        w = self._init_stdev * random.normal(key = random_key, shape = (input_shape[-1], self._n_units))   
        self.weights = w
        return self.weights