# coding: utf-8

from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np


class LogisticRegression(object):
    def __init__(self, input_num):
        X = T.fmatrix('x')
        Y = T.fvector('y')
        W = theano.shared(np.random.randn(input_num))

        p_y_given_x = T.nnet.sigmoid(T.dot(X, W))
        y_pred = p_y_given_x > 0.5

        cost = T.mean((p_y_given_x - Y) ** 2)
        # cost = T.mean(T.nnet.binary_crossentropy(p_y_given_x, Y))
        grad = T.grad(cost=cost, wrt=W)
        update = [[W, W - grad * 0.3]]

        self.train_func = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
        self.predict_func = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

    def train(self, train_x, train_y, nb_epoch=10, batch_size=32):
        train_size = np.array(train_x).shape[0]
        for i in range(nb_epoch):
            for start in range(0, train_size, batch_size):
                cost = self.train_func(train_x[start:start+batch_size], train_y[start:start+batch_size])

            print '[{}]Cost: {}'.format(i, cost)

    def predict(self, test_x):
        y = self.predict_func(test_x)
        return y

def shuffle(x, y):
    index_range = np.arange(x.shape[0])
    np.random.shuffle(index_range)

    return x[index_range], y[index_range]
