import numpy as np
import pandas as pd
from optimizer import Optimizer, GradientDescent


class BaseModel:
    def __init__(self, data):
        self.data = data
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        raise NotImplementedError("The fit method must be implemented by the subclass.")

    def predict(self, X):
        raise NotImplementedError("The predict method must be implemented by the subclass.")

    def evaluate(self, X, y):
        predictions = self.predict(X)
        #mse
        return mse

    def initial_weights(self, n_features):
        self.weights = np.zeros(n_features)
        self.bias = 0.0


class LinearRegression(BaseModel):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def fit(self, X, y):
        self.initial_weights(X.shape[1])
        for i in range(self.optimizer.iterations):
            weights_slope, bias_slope = self.step(X, y)
            self.weights, self.bias = self.optimizer.update(self.weights, self.bias, weights_slope, bias_slope)

    def step(self, X, y):
        n = len(X)
        m = len(self.weights)
        
        weights_slope = np.zeros(m)
        bias_slope = 0

        for i in range(n):
            for j in range(m):    #calc slope using derivative for not free variables
                weights_slope[j] += -2 / n * X[i, j] * (y[i] - (sum(self.weights[k] * X[i, k] for k in range(m)) + self.bias))   
            bias_slope += -2 / n * (y[i] - (sum(self.weights[k] * X[i, k] for k in range(m)) + self.bias))

        return weights_slope, bias_slope
