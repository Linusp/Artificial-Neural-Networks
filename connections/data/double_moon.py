#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def half_moon(center, inner_radius, outer_radius, rotate=0.0):
    res = set([])
    total_num = 0.1 * (outer_radius ** 2 - inner_radius ** 2)
    rotate_rad = np.pi * rotate / 180.0

    for _ in range(int(total_num)):
        rad = np.pi * np.random.random_sample() + rotate_rad
        radius = np.random.randint(inner_radius, outer_radius + 1)

        x = center[0] + radius * np.cos(rad)
        y = center[1] + radius * np.sin(rad)
        res.add((x, y))

    return res


def double_moon(inner_radius=20, outer_radius=40, dis=1, rotate=0.0):
    x_1 = np.random.randint(0, 10)
    y_1 = np.random.randint(0, 10)

    y_2 = y_1 - dis
    x_2 = np.random.randint(x_1 - inner_radius,
                            x_1 + 2 * inner_radius - outer_radius + 1) \
                            + outer_radius

    rotate_rad = np.pi * rotate / 180.0
    x_1, y_1 = x_1 * np.cos(rotate_rad) + y_1 * np.sin(rotate_rad), \
               y_1 * np.cos(rotate_rad) - x_1 * np.sin(rotate_rad)
    x_2, y_2 = x_2 * np.cos(rotate_rad) + y_2 * np.sin(rotate_rad), \
               y_2 * np.cos(rotate_rad) - x_2 * np.sin(rotate_rad)

    res_1 = np.array(list(half_moon((x_1, y_1), inner_radius, outer_radius, rotate)))
    res_class_1 = np.ones((res_1.shape[0], ))
    res_2 = np.array(list(half_moon((x_2, y_2), inner_radius, outer_radius, rotate + 180.0)))
    res_class_2 = np.zeros((res_2.shape[0], ))

    return np.vstack((res_1, res_2)), np.hstack((res_class_1, res_class_2))


def plot_points(data):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    for point in data:
        ax1.plot(point[0], point[1], 'ro')

    plt.savefig('a.png')

if __name__ == '__main__':
    res1, res2 = double_moon(10, 30, dis=-10, rotate=30)
    plot_points(res1)
