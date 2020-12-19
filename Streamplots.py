import numpy as np
import matplotlib.pyplot as plt

U = np.load('Solution_3_U.npy')
V = np.load('Solution_3_V.npy')
p = np.load('Solution_3_p.npy')

M = U.shape[1] - 2

Dx = Dy = 1 / M

# M x M cells + Ghost cells on either boundaries
x = np.linspace(-Dx/2, 1 + Dx/2, M+2)
y = np.linspace(-Dy/2, 1 + Dy/2, M+2)

X, Y = np.meshgrid(x, y, indexing='ij')
plt.figure(figsize=[10, 10])

# plt.contourf(X, Y, np.sqrt(U[-1]**2 + V[-1]**2), 1000)
plt.streamplot(X.T, Y.T, U[-1].T, V[-1].T, 4, arrowstyle='-')
plt.xlim([0,1])
plt.ylim([0,1])

# plt.savefig('Velocity Field.jpeg')
plt.xlabel('x')
plt.ylabel('y')

plt.title(r'$40\times 40$')
plt.show()