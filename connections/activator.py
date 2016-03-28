# coding: utf-8

from __future__ import absolute_import

from connections.utils import (
    sigmoid,
    dsigmoid,
    tanh,
    dtanh,
    ReLU,
    dReLU,
)


class Activator(object):

    @classmethod
    def active(self, x):
        raise NotImplementedError

    @classmethod
    def grad(self, x):
        raise NotImplementedError


class SigmoidActivator(Activator):

    @classmethod
    def active(self, x):
        return sigmoid(x)

    @classmethod
    def grad(self, x):
        return dsigmoid(x)


class TanhActivator(Activator):

    @classmethod
    def active(self, x):
        return tanh(x)

    @classmethod
    def grad(self, x):
        return dtanh(x)


class ReLUActivator(Activator):

    @classmethod
    def active(self, x):
        return ReLU(x)

    @classmethod
    def grad(self, x):
        return dReLU(x)
