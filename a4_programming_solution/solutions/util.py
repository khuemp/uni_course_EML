import pandas as pd
import numpy as np
import scipy as sp
from scipy import linalg

class loess():
    def __init__(self, x, y, weights=None):
        self.x = x
        self.y = y
        self.w = weights
        self.fit()

    def fit(self):
        xh = np.array([np.ones_like(self.x), self.x])
        if self.w is not None:
            W = np.diag(self.w)
            L = xh @ W @ xh.T
            R = self.y @ W @ xh.T
        else:
            L = xh @ xh.T
            R = self.y @ xh.T
        self.params = linalg.solve(L, R)

    def predict(self, x0):
        xh = np.array([np.ones_like(x0), x0])
        return self.params @ xh
