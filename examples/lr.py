# coding: utf-8

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from materials import double_moon
from connections.logistic import LogisticRegression


if __name__ == '__main__':
    lr = LogisticRegression(2)

    double_moon = double_moon(50, 80, 1)
    train_x = double_moon[['x', 'y']].as_matrix()
    train_y = double_moon['class'].as_matrix()

    lr.train(train_x, train_y)
