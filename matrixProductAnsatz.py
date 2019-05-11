import numpy as np
from numpy.linalg import svd
from numpy.linalg import multi_dot
from useful import bits

# Problem size
N = 4
DIM = 2**N

# Initialize quantum state
psi = np.load('data/groundState' + str(N) + '.npy')
print(psi)

# Create containers
coefficients = []
matrices = []

# Do the MPA
M = 1
for j in range(1, N):

    # Make SVD
    psi = psi.reshape((2*M), 2**(N-j))
    u, d, v = svd(psi, full_matrices=False)

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
    multi_dot([matrices[n][b] for n, b in enumerate(bits(s, N))])
    for s in range(DIM) ])
print()
print(psi)
