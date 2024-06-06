import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_regression_line2D(X, y, weights, bias):
    plt.figure(figsize=(8, 6))

    # Rysowanie punktów danych
    plt.scatter(X, y, color='blue', label='Data Points')

    # Tworzenie linii regresji
    x_line = np.linspace(np.min(X), np.max(X), 100)
    y_line = weights * x_line + bias
    plt.plot(x_line, y_line, color='red', label='Regression Line')

    plt.title('Linear Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_regression_line3D(X, y, weights, bias):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Rysowanie punktów danych
    ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data Points')

    # Tworzenie linii regresji
    x_line = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y_line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    x_line, y_line = np.meshgrid(x_line, y_line)
    z_line = weights[0] * x_line + weights[1] * y_line + bias
    ax.plot_surface(x_line, y_line, z_line, color='red', alpha=0.5, label='Regression Plane')

    ax.set_title('Linear Regression in 3D')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('y')
    ax.legend()
    ax.grid(True)

    plt.show()