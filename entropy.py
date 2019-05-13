# Calculates MPA-s and plots entropies

import numpy as np
import numpy.linalg as la
from numpy.random import normal
from matplotlib import pyplot as plt
from useful import bits, mpaEntropy

ground = False
periodic = False
chains = range(3, 21, 2)

entropies = []
for n in chains:
    entropies.append(mpaEntropy(n, ground, periodic))

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.set_ylabel('$E(| \\psi_0 \\rangle)$')
ax.set_xlabel('$j/n$')

for n, c in enumerate(chains):
    ax.plot(np.linspace(0, 1, len(entropies[n])), entropies[n])
if min(chains) % 2 == 0:
    ax.set_title('Sodi $n$ od $%d$ do $%d$' % (min(chains), max(chains)))
else:
    ax.set_title('Lihi $n$ od $%d$ do $%d$' % (min(chains), max(chains)))
plt.show()
