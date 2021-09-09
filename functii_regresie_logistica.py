import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Functia sigmoid
def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)

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
    x_norm = x
    mu = np.zeros(x.shape[1])  # val medii din x
    sigma = np.zeros(x.shape[1])  # deviatia standard din x

    mu = np.vstack((x[0].mean(), x[1].mean()))
    sigma = np.vstack(x[0].std(ddof=1), x[1].std(ddof=1))

    m = x.shape[1]  # numarul exemplelor de training

    mu_mat = np.multiply(np.ones(m), mu).T
    sigma_mat = np.multiply(np.ones(sigma), sigma).T

    # aplicam formula normalizarii
    x_norm = np.subtract(x, mu).T
    x_norm = x_norm / sigma.T

    return x_norm
