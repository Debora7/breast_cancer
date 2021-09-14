import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functii_regresie_logistica as frl

# Citirea datelor
data = pd.read_csv('data.csv')
x = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']].values
y = data[['diagnosis']].values
m, n = x.shape
# print(x.shape, y.shape)

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
test_theta = np.array([13, 5, 2, 11, 1])
cost = frl.cost(x, y, test_theta)
print('Cu parametrii theta = [%d, %d, %d, %d, %d]'
      '\nEroarea calculata = %.3f'
      % (test_theta[0], test_theta[1], test_theta[2], test_theta[3], test_theta[4], cost))

# Initializam parametrii modelului cu zero
initial_theta = np.zeros(n+1)

# Normalizarea datelor
# x = frl.normalizare(x)  # 40%
x = frl.norm(x)  # 60%

# Setarile algoritmului gradient descent
iterations = 1000
alpha = 0.001
theta, J_history, theta_history = frl.grad_desc(x, y, initial_theta, alpha, iterations)
print('Parametrii theta obtinuti cu gradient descent: {:.4f}, {:.4f}'.format(*theta))

# Realizarea predictiei
vct = [0.702, 0.23, 0.543, 0.4637, 1]
prob = frl.sigmoid(np.dot(vct, theta))
print('Pentru parametrii %.3f, %.3f, %.3f, %.3f, %.3f, pobabilitatea de boala este: %d'
      % (vct[0], vct[1], vct[2], vct[3], vct[4], prob))

# Afisarea functiei de eroare
frl.plotConverge(J_history)

# Acuratetea pe setul de date
p = frl.predict(theta, x)
print('Acuratetea pe setul de antrenare: {:.2f} %'.format(np.mean(p == y) * 100))
