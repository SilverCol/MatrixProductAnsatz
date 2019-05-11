import numpy as np
import numpy.linalg as la
from numpy.random import normal
from useful import bits

# Problem parameters
N = 12
DIM = 2**N
ground = True
periodic = False

# Initialize quantum state
if not ground:
    psi = np.array([ [normal()] for n in range(DIM) ])
    psi /= la.norm(psi)
elif periodic: psi = np.load('data/groundState' + str(N) + 'p.npy')
else: psi = np.load('data/groundState' + str(N) + '.npy')
initial = psi

# Create containers
coefficients = []
matrices = []

# Do the MPA
M = 1
for j in range(1, N):

    # Make SVD
    psi = psi.reshape((2*M), 2**(N-j))
    u, d, v = la.svd(psi, full_matrices=False)

    # Store results
    coefficients.append(d)
    matrices.append([u[::2], u[1::2]])

    # Prepare for next iteration
    psi = np.matmul(np.diag(d), v)
    M = d.size
# Final MPA step
matrices.append([psi[:, 0], psi[:, 1]])

# Reconstruct initial state
psi = np.array([
    la.multi_dot([matrices[n][b] for n, b in enumerate(bits(s, N))])
    for s in range(DIM) ])
print('Max. error: %.2e' % np.amax(psi - initial))
