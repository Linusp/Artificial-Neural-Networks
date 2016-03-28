# coding: utf-8

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1.0 - tanh(x) * tanh(x)


def ReLU(x):
    return np.maximum(x, 0)


def dReLU(x):
    def _drelu_on_single(x):
        return 1 if x >= 0 else 0

    vfunc = np.vectorize(_drelu_on_single)
    return vfunc(x)
