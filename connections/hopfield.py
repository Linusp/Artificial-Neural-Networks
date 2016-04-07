# coding: utf-8

import numpy as np


class HopfieldNetwork(object):
    def __init__(self, n):
        self.connections = np.zeros((n, n), dtype=int)
        self.values = np.zeros((n, ), dtype=int)

    def train(self, patterns):
        """每个 pattern 都是由 0/1 组成的向量，大小与 n 相等"""
        for p in patterns:
            for i in range(len(p)):
                for j in range(len(p)):
                    if i == j:
                        continue

                    self.connections[i, j] += (2 * p[i] - 1) * (2 * p[j] - 1)

    def recog(self, partial_pattern):
        """根据不完整的模式来进行重建"""
        self.values = partial_pattern
        update_orders = range(len(self.values))
        np.random.shuffle(update_orders)

        changed_num = -1
        min_changed_num = 5
        loop_cnt = 0
        while changed_num < 0 or changed_num >= min_changed_num:
            changed_num = 0
            for index in update_orders:
                old_val = self.values[index]
                if np.dot(self.connections[index], self.values) >= 0:
                    self.values[index] = 1
                else:
                    self.values[index] = 0

                if old_val != self.values[index]:
                    changed_num += 1

            print '{0} values changed this pass'.format(changed_num)
            loop_cnt += 1

        print 'end with {0} loop'.format(loop_cnt)
        return self.values


if __name__ == '__main__':
    hp = HopfieldNetwork(64)

    number_two = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    partial_number_two = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    number_eight = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)
    partial_number_eight = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    hp.train([number_two.flatten(), number_eight.flatten()])
    print hp.connections
    v = hp.recog(np.zeros((64, )))
    print v.reshape((8, 8))
