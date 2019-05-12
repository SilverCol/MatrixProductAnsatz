import numpy as np
import numpy.linalg as la
from numpy.random import normal
from matplotlib import pyplot as plt
from useful import bits, mpaEntropy

ground = False
periodic = False
chains = range(2, 19, 2)

entropies = []
for n in chains:
    entropies.append(mpaEntropy(n, ground, periodic))

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
for n in range(len(chains)):
    ax.plot(np.linspace(0, 1, len(entropies[n])), entropies[n])
plt.show()
