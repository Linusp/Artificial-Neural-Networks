# coding: utf-8

from __future__ import absolute_import

import numpy as np
from connections.activator import SigmoidActivator


class LogisticRegression(object):
    def __init__(self, input_num):
        self.weights = np.ones((input_num + 1, ))
        self.activator = SigmoidActivator

    def forward(self, data):
        x_data = data
        if len(x_data.shape) == 1:
            x_data = x_data.reshape((1, data.shape[0]))

        self.input_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))
        self.output_data = self.activator.active(self.input_data.dot(self.weights))
        return self.output_data

    def predict(self, data):
        return 1 if self.forward(data) > 0.5 else 0

    def error(self, target):
        error = self.output_data - target
        return error

    def backward(self, target, learning_rate=0.1, weight_penalty=0.3):
        p_y_given_x = self.output_data
        weights_decay = learning_rate * weight_penalty * self.weights
        weights_diff = learning_rate * (p_y_given_x - target).reshape((target.shape[0], 1)) * self.input_data
        weights_diff = (weights_diff.sum(axis=0) + weights_decay) / target.shape[0]
        self.weights -= weights_diff

    def train(self, train_data, test_data, learning_rate=0.3, weight_penalty=0.1):
        train_data_x, train_data_y = train_data
        test_data_x, test_data_y = test_data

        for i in range(20):
            self.forward(test_data_x)
            error = np.sqrt((self.error(test_data_y) ** 2).mean())
            print 'Train loop {0}: test error is {1}'.format(i + 1, error)

            self.forward(train_data_x)
            self.backward(train_data_y, learning_rate, weight_penalty)


def shuffle(x, y):
    index_range = np.arange(x.shape[0])
    np.random.shuffle(index_range)

    return x[index_range], y[index_range]


if __name__ == '__main__':
    from connections.data.double_moon import double_moon
    res1, res2 = double_moon(inner_radius=10, outer_radius=30, dis=-3, rotate=30)
    res1, res2 = shuffle(res1, res2)

    train_size = len(res1) * 9 / 10
    lr = LogisticRegression(2)
    lr.train((res1[:train_size], res2[:train_size]), (res1[train_size:], res2[train_size:]))
