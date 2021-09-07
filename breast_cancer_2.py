import numpy
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op


# Functia signoid
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))

    return g


# Functia cost
def cost(theta, x, y):
    n, m = x.shape
    h = x.dot(theta)

    J = -(np.sum(y * np.log(sigmoid(h)) + (1 - y) * np.log(1 - sigmoid(h))) / m)

    return J


# Functia gradient descendent
def grad_desc(theta, x, y):
    m, n = x.shape
    h = np.dot(theta)
    eroare = sigmoid(h) - y
    grad = 1 / m * (x.T).dot(eroare)

    return grad


# Functia de calcul a aacurateti
def acuratete(theta, x):
    p = sigmoid(x.dot(theta)) >= 0.5

    return p


# Functie de normalizare a datelor
def normalizare(x):
    x_norm = x
    mu = np.zeros(x.shape[1])
    sigma = np.zeros(x.shape[1])

    mu = np.vstack((x[0].mean(), x[1].mean(), x[2].mean()))
    sigma = np.vstack((x[0].std(ddof=1), x[1].std(ddof=1), x[2].std(ddof=1)))

    m = x.shape[1]

    mu_mat = np.multiply(np.ones(m), mu).T
    sigma_mat = np.multiply(np.ones(m), sigma).T

    x_norm = np.subtract(x, mu).T
    x_norm = x_norm / sigma.T

    return [x_norm, mu, sigma]


# Citirea datelor
data = pd.read_csv('data.csv')
x = np.vstack(np.asarray(data.radius_mean.values),
              np.asarray(data.texture_mean.values),
              np.asarray(data.perimeter_mean.values),
              np.asarray(data.area_mean.values),
              np.asarray(data.smoothness_mean.values))
y = np.asarray(data.diagnosis.values)

normaliz_rezult = normalizare(x)
x = np.asarray(normaliz_rezult[0]).T

mu = normaliz_rezult[1]
sigma = normaliz_rezult[2]

m = len(y)
n = len(x)

x = np.vstack((np.ones(m), x)).T

m, n = x.shape
initial_theta = np.zeros(n)
rezultat = op.minimize(fun = cost,
                                 x0 = initial_theta,
                                 args = (x, y),
                                 method = 'BFGS',
                                 jac = grad_desc)
theta = rezultat.x
mesaj = rezultat.message

query = np.asarray([1, 5.00, 1.10, 0.4])

# Scale and Normalize the query
query_Normalized = np.asarray([1, ((query[1]-float(mu[0]))/float(sigma[0])),\
                               ((query[2]-float(mu[1]))/float(sigma[1])),\
                               ((query[3]-float(mu[2]))/float(sigma[2]))])

# Calculate the prediction using the Logistic Function
prediction = sigmoid(query_Normalized.dot(theta));

# Calculate accuracy
p = acuratete(theta, x)
p = (p == y) * 100

# Print the output
print("Your Query is: "+str(query.tolist()[1:])[1:-1])
print("BFGS Message: "+str(mesaj))
print("Theta found: "+str(theta)[1:-1])
print("Train Accuracy: "+str(p.mean()))
print("Prediction: "+str(prediction))