# coding: utf-8

import numpy as np

from connections.models import MLP
from connections.layers import DenseLayer


m = MLP()
m.add_layer(DenseLayer(3, 4))
m.add_layer(DenseLayer(4, 2, activation='sigmoid'))
m.compile()

x = np.random.randn(4, 3)
print m.predict(x)
