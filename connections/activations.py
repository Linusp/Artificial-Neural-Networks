# coding: utf-8

from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np


def get_activation(name):
    if name == 'linear':
        return lambda x: x
    elif name == 'sigmoid':
        return T.nnet.sigmoid
    elif name == 'softmax':
        return T.nnet.softmax
    else:
        raise NotImplementedError
