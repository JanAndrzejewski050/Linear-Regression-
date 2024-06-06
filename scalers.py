import numpy as np
from metrics import Metrics

class MinMaxScaler:
    def __init__(self):
        self.min_x = None
        self.max_x = None

    def fit(self, X):
        self.min_x = X.min(axis=0)
        self.max_x = X.max(axis=0)

    def transform(self, X):
        return (X - self.min_x) / (self.max_x - self.min_x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.standard_deviation = None

    def fit(self, X):
        self.mean = Metrics.mean(X)
        self.standard_deviation = Metrics.standard_deviation(X)

    def transform(self, X):
        return (X - self.mean) / self.standard_deviation
