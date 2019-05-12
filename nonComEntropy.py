
import numpy as np
from matplotlib import pyplot as plt
from useful import bits, svdEntropy

pbc = []
obc = []

for n in range(2, 21):
    pbc.append(svdEntropy(n, 1, True))
    obc.append(svdEntropy(n, 1, False))

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.plot(np.linspace(2, 20, 19), pbc)
ax.plot(np.linspace(2, 20, 19), obc)
plt.show()
