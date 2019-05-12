# Calculates entropies for different granulations of non-compact
# 20-chain bipartition

import numpy as np
from matplotlib import pyplot as plt
from useful import bits, svdEntropy

pbc = []
obc = []

domain = (1, 2, 5, 10)

for n in domain:
    pbc.append(svdEntropy(20, n, True))
    obc.append(svdEntropy(20, n, False))

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.set_ylabel('$E(| \\psi_0 \\rangle)$')
ax.set_xlabel('Perioda biparticije')
ax.set_title('$n = 20$')
pline, = ax.plot(domain, pbc)
oline, = ax.plot(domain, obc)
ax.legend((pline, oline), ('PRP', 'ORP'))
plt.show()
