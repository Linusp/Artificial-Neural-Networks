# coding: utf-8

from __future__ import absolute_import

import theano
import theano.tensor as T
import numpy as np


ACTIVATIONS = {
    'linear': lambda x: x,
    'sigmoid': T.nnet.sigmoid,
    'tanh': T.tanh,
    'softmax': T.nnet.softmax,
}


def get_activation(name):
    act_func = ACTIVATIONS.get(name)
    if not act_func:
        raise NotImplementedError(name)

    return act_func
