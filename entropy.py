import numpy as np
import numpy.linalg as la
from numpy.random import normal
from matplotlib import pyplot as plt
from useful import bits, mpa

# Problem parameters
N = 18
DIM = 2**N
ground = False
periodic = False

# Initialize quantum state
if not ground:
    psi = np.array([ [normal()] for n in range(DIM) ])
    psi /= la.norm(psi)
elif periodic: psi = np.load('data/groundState' + str(N) + 'p.npy')
else: psi = np.load('data/groundState' + str(N) + '.npy')
initial = psi

# Cacluate the MPA
print('Calculating MPA...')
A, lambdas = mpa(psi)

# Reconstruct initial state for error estimation
#print('Reconstructing initial state...', end=' ', flush=True)
#psi = np.array([
#    la.multi_dot([A[n][b] for n, b in enumerate(bits(s, N))])
#    for s in range(DIM) ])
#print('Done')
#print('Max. error: %.2e' % np.amax(psi - initial))

# Calculate entropy
entropies = [-2 * np.sum(a**2 * np.log(a)) for a in lambdas]

plt.rcParams.update({'font.size': 15})
fig = plt.figure()
ax = fig.subplots()

ax.grid()
ax.plot(np.linspace(1, N-1, N-1), entropies)
plt.show()
