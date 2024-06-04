import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data.csv')

def plot_3d_regression(data, factorsList):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['mouse_size'], data['tail_length'], data['mouse_weight'], color='blue')

    x_surf, y_surf = np.meshgrid(np.linspace(data['mouse_size'].min(), data['mouse_size'].max(), 100),
                                 np.linspace(data['tail_length'].min(), data['tail_length'].max(), 100))
    z_surf = factorsList[0] + factorsList[1] * x_surf + factorsList[2] * y_surf

    ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

    ax.set_xlabel('Mouse Size')
    ax.set_ylabel('Tail Lenght')
    ax.set_zlabel('Mouse Weight')

    plt.show()

def calc_mean(data):
    mean = 0
    n = len(data)
    for i in range(n):
        yi = data.iloc[i].mouse_weight
        mean += yi
    return mean/n

def calc_r_squared(m, b, data):
    ss_fit = 0; ss_mean = 0; mean = calc_mean(data); n = len(data)
    for i in range(n):
        xi = data.iloc[i].mouse_size
        yi = data.iloc[i].mouse_weight
        ss_fit += (yi - (m*xi + b)) ** 2
        ss_mean += (yi - mean) ** 2
    r_squared = (ss_mean - ss_fit) / ss_mean
    return r_squared


def gradient_descent(factorsList, L, data):
    n = len(data); m = len(factorsList); slopeList = [0 for i in range(m)]; xsList = []     # xsList = variables list

    for i in range(n):
        y_i = data.iloc[i].mouse_weight
        for j in range(m):      # getting all variables
            xsList.append(data.iloc[i][j])

        for j in range(m-1):    #calc slope using derivative for not free variables
            slopeList[j] += -2 / n * xsList[j] * (y_i - (sum(factorsList[k] * xsList[k] for k in range(m))))   
        slopeList[m-1] += -2 / n * (y_i - (sum(factorsList[k] * xsList[k] for k in range(m))))

    for i in range(m):
        factorsList[i] = factorsList[i] - L * slopeList[i]

    return factorsList


def linear_regression(L, data):
    factorsList = [0 for i in range(len(data.iloc[0]))]
    for i in range(iterations):
        factorsList = gradient_descent(factorsList, learning_rate, data)

    print(factorsList)
    #print(f"R_squared = {calc_r_squared(m, b, data)}")

    plot_3d_regression(data, factorsList)


learning_rate = 0.001; iterations = 1000

linear_regression(learning_rate, data)

