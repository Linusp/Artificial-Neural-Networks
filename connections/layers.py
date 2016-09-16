# coding: utf-8

from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np

from connections.activations import get_activation


class Layer(object):
    def __init__(self, input_dim, output_dim, activation='linear'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = get_activation(activation)

        self.params = []

    def get_params(self):
        return self.params

    def get_output_for(self, x):
        raise NotImplementedError


class DenseLayer(Layer):
    """经典的全连接层"""
    def __init__(self, input_dim, output_dim, activation='linear'):
        super(DenseLayer, self).__init__(input_dim, output_dim, activation)

        W = theano.shared(np.random.randn(input_dim, output_dim), name='W')
        b = theano.shared(np.zeros((output_dim,)), name='b')

        self.weights = W
        self.bias = b
        self.params = [self.weights, self.bias]

    def __repr__(self):
        return '<DenseLayer({}, {})>'.format(input_dim, output_dim)

    def get_output_for(self, x):
        return self.activation(T.dot(x, self.weights) + self.bias)


class RecurrentLayer(Layer):
    def __init__(self, input_dim, output_dim, activation='linear', return_sequences=False):
        super(RecurrentLayer, self).__init__(input_dim, output_dim, activation)

        W = theano.shared(np.random.randn(input_dim, output_dim), name='W')
        U = theano.shared(np.random.randn(output_dim, output_dim), name='U')
        b = theano.shared(np.zeros((output_dim,)), name='b')

        self.W = W
        self.U = U
        self.b = b
        self.params = [self.W, self.U, self.b]
        self.return_sequences = return_sequences

    def __repr__(self):
        return '<RecurrentLayer({}, {})>'.format(self.input_dim, self.output_dim)

    def get_output_for_timestep(self, x, state):
        """for a timestep"""
        return self.activation(T.dot(x, self.W) + T.dot(state, self.U) + self.b)

    def get_output_for(self, x):
        """for a sequence"""
        state = theano.shared(np.zeros(self.output_dim,))
        y, _ = theano.scan(
            self.get_output_for_timestep,
            sequences=x,
            outputs_info=[state]
        )
        return y if self.return_sequences else y[-1]
