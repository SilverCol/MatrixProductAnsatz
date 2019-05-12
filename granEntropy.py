
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
ax.plot(domain, pbc)
ax.plot(domain, obc)
plt.show()
