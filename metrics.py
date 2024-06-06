import numpy as np
import pandas as pd 

class Metrics:
    @staticmethod
    def mean_dependent_y(data):
        return np.mean(data.iloc[:, -1])

    @staticmethod
    def mean(data):
        return np.mean(data)

    @staticmethod
    def variance(data):
        data = np.array(data, dtype=float)  # Konwersja danych na typ float
        var = 0
        n = len(data)
        mean_value = Metrics.mean(data)
        for v in data:
            var += (v - mean_value) ** 2
        return var / n

    @staticmethod
    def standard_deviation(data):
        return np.sqrt(Metrics.variance(data))

    @staticmethod
    def mean_squared_error(data):
        ss_mean = 0; n = len(data)
        for i in range(len(data)):
            ss_mean += (dependent_y[i] - Metrics.mean_dependent_y(data)) ** 2
        return ss_mean

    @staticmethod
    def fit_squared_error(data, factors_list):
        dependent_y = data.iloc[:, -1]
        independent_xs = data.iloc[:, :-1]
        ss_fit = 0; n = len(data); m = len(factors_list)
        for i in range(n):
            pred_y = 0
            for j in range(m-1):
                pred_y += factors_list[j] * independent_xs.iloc[i, j]
            ss_fit += (dependent_y.iloc[i] - pred_y) ** 2
        return ss_fit

    @staticmethod
    def r_squared(data, factors_list):
        return (Metrics.mean_squared_error(data) - Metrics.fit_squared_error(data, factors_list)) / Metrics.mean_squared_error(data)
