# Calculates entropy for simmetric bipartition

import numpy as np
from matplotlib import pyplot as plt
from useful import bits, svdEntropy

pbc = []
obc = []

for n in range(1, 11):
    pbc.append(svdEntropy(2*n, n, True))
    obc.append(svdEntropy(2*n, n, False))

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.set_ylabel('$E(| \\psi_0 \\rangle)$')
ax.set_xlabel('$n$')
pline, = ax.plot(np.linspace(2, 20, 10), pbc)
oline, = ax.plot(np.linspace(2, 20, 10), obc)
ax.legend((pline, oline), ('PRP', 'ORP'))
plt.show()
