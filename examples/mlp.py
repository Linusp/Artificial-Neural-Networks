# coding: utf-8

import numpy as np

from materials import double_moon
from connections.models import Model
from connections.layers import DenseLayer


if __name__ == '__main__':
    m = Model()
    m.add_layer(DenseLayer(2, 5, activation='sigmoid'))
    m.add_layer(DenseLayer(5, 1, activation='sigmoid'))
    m.compile(lr=0.1)

    double_moon = double_moon(50, 80, 1)
    train_x = double_moon[['x', 'y']].as_matrix()
    train_y = double_moon['class'].as_matrix()
    train_y = train_y.reshape((train_y.shape[0], 1))

    m.train(train_x, train_y)
