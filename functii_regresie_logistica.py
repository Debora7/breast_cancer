import numpy as np
import matplotlib.pyplot as plt


# Functia sigmoid
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)
    # print(z)
    g = 1 / (1 + np.exp(-z))

    return g


# Functia de eroare
def eroare(theta, x):
    h = sigmoid(np.dot(x, theta))

    return h


# Functia de cost
def cost(x, y, theta):
    m = y.size
    h = eroare(theta, x)

    J = -(np.sum(y * np.log(sigmoid(h)) + (1 - y) * np.log(1 - sigmoid(h))) / m)

    return J


# Functia gradient descendent
def grad_desc(x, y, theta, alpha, nr_items):
    m = y.size
    theta = theta.copy()
    J_vechi = []
    theta_vechi = []

    for i in range(nr_items):
        theta_vechi.append(list(theta))

        h = eroare(theta, x)
        theta[0] = theta[0] - (alpha/m) * (np.sum(h-y))
        theta[1] = theta[1] - (alpha/m) * (np.sum((h-y) * x[:, 1]))

        J_vechi.append(cost(x, y, theta))

    return theta, J_vechi, theta_vechi


# Functia de predictie
def predict(theta, x):
    p = eroare(theta, x)
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p


# Functie de normalizare a datelor
def normalizare(x):
    std = np.std(x)
    mean = np.mean(x)

    x_norm = (x - mean) / std

    return x_norm


# Functie de afisare a erorii
def plotConverge(J_history):
    plt.plot(range(len(J_history)), J_history, 'bo')
    plt.title('Convergenta functiei eroare')
    plt.xlabel('Numarul de iteratii')
    plt.ylabel('Functia eroare')
    plt.show()
