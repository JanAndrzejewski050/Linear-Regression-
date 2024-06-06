import numpy as np
import pandas as pd
from data_prep import Data_Scale
from metrics import Metrics
from linear_regression import BaseModel, LinearRegression
from optimizer import GradientDescent  
from plot_regression import plot_regression_line2D, plot_regression_line3D


def generate_data(n_samples, n_features):
    # Generowanie przykładowych danych do testów
    X = np.random.rand(n_samples, n_features)
    true_weights = np.random.rand(n_features)
    true_bias = np.random.rand()

    y = np.dot(X, true_weights) + true_bias + np.random.randn(n_samples) * 0.1  # Dodajemy szum

    return X, y


def main():
    # Parametry testowe
    n_samples = 100
    n_features = 2
    learning_rate = 0.01
    iterations = 1000

    # Generowanie danych testowych
    X, y = generate_data(n_samples, n_features)

    # Tworzenie instancji modelu i optymalizatora
    model = LinearRegression(optimizer=GradientDescent(learning_rate=learning_rate, iterations=iterations))

    # Dopasowanie modelu do danych treningowych
    model.fit(X, y)

    # Wyświetlenie wagi i biasu (opcjonalne)
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")

    if n_features == 1:
        plot_regression_line2D(X, y, model.weights, model.bias)
    elif n_features == 2:
        plot_regression_line3D(X, y, model.weights, model.bias)
    else:
        print("to much")


if __name__ == "__main__":
    main()