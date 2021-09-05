import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Incarcarea datelor
data = pd.read_csv('data.csv', sep=",")
x = data[['area_mean']]
y = data[['diagnosis']]
# y_orig = data[['diagnosis']]
# y = y_orig
# y[y == 'M'] = 1
# y[y == 'B'] = 0


# Functia sigmoid
def sigmoid(z):
    z = np.array(z)
    q = np.zeros(z.shape)

    q = 1 / (1 + np.exp(-z))

    return q


z = 0
g = sigmoid(z)
# print('g(',z,') =', g)

# Adaugam o coloana cu valori 1 matricei x
m, n = x.shape
x = np.concatenate([np.ones((m, 1)), x], axis=1)


# Functia eroare
def eroare(theta, x):
    h = sigmoid(np.dot(x, theta))

    return h


# Functia cost
def cost(theta, x, y):
    m = y.size
    h = eroare(theta, x)

    J = -(np.sum(y * np.log(sigmoid(h)) + (1 - y) * np.log(1 - sigmoid(h)))/m)

    return J


# Functia gradient descendent
def calcGradient(theta, X, y):
    m, n = X.shape
    h = X.dot(theta)
    error = sigmoid(h) - y

    gradient = 1 / m * (X.T).dot(error)

    return gradient


# Scalare si normalizare
def FeatureScalingNormalization(X):
    X_norm = X
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.vstack((X[0].mean(), X[1].mean()))

    sigma = np.vstack((X[0].std(ddof=1), X[1].std(ddof=1)))

    m = X.shape[1]

    mu_matrix = np.multiply(np.ones(m), mu).T
    sigma_matrix = np.multiply(np.ones(m), sigma).T

    X_norm = np.subtract(X, mu).T
    X_norm = X_norm / sigma.T

    return [X_norm, mu, sigma]


# Functia pentru scalarea acuratetei
def predict(theta, x):
    p = eroare(theta, x)
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    return p


# Afisarea datelor
for i in range(len(y)):
    if y[i] == 'M':
        plt.plot(x[i, 0], x[i, 1], 'k+')
        i += 1
    else:
        plt.plot(x[i, 0], x[i, 1], 'yo')
        i += 1
    plt.show()

plt.xlabel(data.columns[data.columns.get_loc("Area min")])
plt.ylabel(data.columns[data.columns.get_loc("Diagnosis")])


# Testarea
initial_theta = np.zeros(n+1)
theta = [-24, 1.2, 0.2]

print("J", cost(theta=initial_theta, X=x, y=y))
print("grad", calcGradient(theta=initial_theta, X=x, y=y))

# Afisarea marginii de separatie (decision boundary)
boundary_xs = np.array([np.min(x[:, 1]), np.max(x[:, 1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)

# Calcularea acuratetei
p = predict(theta, x)

# Realizarea predictiei
prob = sigmoid(np.dot([1, 45, 85], theta))

print('Pentru un student cu scorurile 45 si 85,'
      'probabilitatea de admitere prezisa {:.3f}'.format(prob))
print('Valoarea asteptata: 0.687 +/- 0.002\n')
