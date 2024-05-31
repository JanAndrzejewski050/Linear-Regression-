import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

def find_m_and_b(m, b, L, data):
    m_derivative_in_m = 0
    b_derivative_in_b = 0
    for i in range(len(data)):
        xi = data.iloc[i].mouse_size
        yi = data.iloc[i].mouse_weight
        m_derivative_in_m += -2 * xi * (yi - m*xi - b)
        b_derivative_in_b += -2 * (yi - m*xi - b)
    m = m - m_derivative_in_m*L
    b = b - b_derivative_in_b*L
    return m, b

def calc_mean(data):
    mean = 0
    n = len(data)
    for i in range(n):
        yi = data.iloc[i].mouse_weight
        mean += yi
    return mean/n

def calc_r_squared(m, b, data):
    ss_fit = 0
    ss_mean = 0
    mean = calc_mean(data)
    n = len(data)
    for i in range(n):
        xi = data.iloc[i].mouse_size
        yi = data.iloc[i].mouse_weight
        ss_fit += (yi - (m*xi + b)) ** 2
        ss_mean += (yi - mean) ** 2
    r_squared = (ss_mean - ss_fit) / ss_mean
    return r_squared


m = 0; b = 0; learning_rate = 0.0001; iterations = 1000

for i in range(iterations):
    # if i%(iterations//6) == 0: 
    #     print(f"Loading: {round(i/iterations*100, 2)}%")
    m, b = find_m_and_b(m, b, learning_rate, data)

print(f"m: {m}, b: {b}")
print(f"R_squared = {calc_r_squared(m, b, data)}")

plt.plot(list(range(10, 38)), [m*x + b for x in range(10, 38)], color = "red")
plt.scatter(data['mouse_size'], data['mouse_weight'])
plt.show()
