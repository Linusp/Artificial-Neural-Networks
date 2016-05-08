# coding: utf-8

from __future__ import absolute_import

from itertools import chain

import theano
import theano.tensor as T
import numpy as np


class MLP(object):
    def __init__(self, layers=None):
        self.layers = [] if not layers else layers
        self.params = list(chain.from_iterable([layer.params for layer in self.layers]))

    def add_layer(self, layer):
        if self.layers and self.layers[-1].output_dim != layer.input_dim:
            raise ValueError('layer\'s input_dim mismatched, expect {} but get'.format(self.layers[-1].output_dim, layer.input_dim))
        self.layers.append(layer)
        self.params.extend(layer.params)

    def compile(self, lr=0.1):
        X = T.fmatrix('x')
        Y = T.fmatrix('y')

        funcs = [layer.get_output_for for layer in self.layers]
        p_y_given_x = reduce(lambda x, fn: fn(x), [X] + funcs)
        self.predict_func = theano.function(
            inputs=[X],
            outputs=p_y_given_x,
            allow_input_downcast=True
        )

        cost = T.mean((p_y_given_x - Y) ** 2)
        grad = [T.grad(cost, param) for param in self.params]
        updates = [
            (param, param - lr * param)
            for param, gparam in zip(self.params, grad)
        ]
        self.train_func = theano.function(
            inputs=[X, Y],
            outputs=cost,
            updates=updates,
            allow_input_downcast=True
        )

    def predict(self, x):
        if not self.predict_func:
            raise Exception('should compile model')

        return self.predict_func(x)

    def train(self, train_x, train_y, nb_epoch=10, batch_size=32):
        train_size = np.array(train_x).shape[0]
        print 'train on {} record'.format(train_size)
        for i in range(nb_epoch):
            for start in range(0, train_size, batch_size):
                batch_x = train_x[start:start+batch_size]
                batch_y = train_y[start:start+batch_size]
                cost = self.train_func(batch_x, batch_y)

            print '[{}]Cost: {}'.format(i, cost)
