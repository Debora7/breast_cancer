import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functii_regresie_logistica as frl

# Citirea datelor
data = pd.read_csv('data.csv')
x = data[['area_mean']].values
y = data[['diagnosis']].values
m, n = x.shape
# print(x.shape, y.shape)

# Afisarea datelor
# malign
plt.hist(x[y == 'M'], bins=30)
plt.xlabel("Aria medie - Malign")
plt.ylabel("Frecventa")
# plt.show()

# bening
plt.hist(x[y == 'B'], bins=30)
plt.xlabel("Aria medie - Belign")
plt.ylabel("Frecventa")
# plt.show()

# Functia sigmoid
z = 0
g = frl.sigmoid(z)
print('g(', z, ') =', g)

# Adaugam o coloana cu valori de 1 matricei x.
x = np.concatenate([np.ones((m, 1)), x], axis=1)
# print(x.shape)

# Modificarea valorilor lui y
y[y == 'M'] = 1
y[y == 'B'] = 0

# Calcularea erorii
test_theta = np.array([13, 5])
cost = frl.cost(x, y, test_theta)
print('Cu parametrii theta = [%d, %d] \nEroarea calculata = %.3f' % (test_theta[0], test_theta[1], cost))

# Initializam parametrii modelului cu zero
initial_theta = np.zeros(n+1)

# Setarile algoritmului gradient descent
iterations = 1000000
alpha = 0.001
theta, J_history, theta_history = frl.grad_desc(x, y, initial_theta, alpha, iterations)
print('Parametrii theta obtinuti cu gradient descent: {:.4f}, {:.4f}, {:.4f}'.format(*theta))