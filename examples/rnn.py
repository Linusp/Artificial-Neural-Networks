# coding: utf-8

import numpy as np

from connections.models import Model
from connections.layers import DenseLayer, RecurrentLayer


if __name__ == '__main__':
    m = Model()
    m.add_layer(RecurrentLayer(2, 5, activation='tanh', return_sequences=True))
    m.add_layer(DenseLayer(5, 1, activation='sigmoid'))
    m.compile(lr=0.1)

    print m.predict(np.arange(6).reshape(3, 2))
