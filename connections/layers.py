# coding: utf-8

from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np

from connections.activations import get_activation


class DenseLayer(object):
    """经典的全连接层"""
    def __init__(self, input_dim, output_dim, activation='linear'):
        self.input_dim = input_dim
        self.output_dim = output_dim

        W = theano.shared(np.random.randn(input_dim, output_dim), name='W')
        b = theano.shared(np.zeros((output_dim,)), name='b')

        self.weights = W
        self.bias = b
        self.params = [self.weights, self.bias]

        self.activation = get_activation(activation)

    def get_params(self):
        return self.params

    def get_output_for(self, x):
        return self.activation(T.dot(x, self.weights) + self.bias)
