import numpy as np
from numpy.linalg import svd
from useful import bits

# Parameters
N = 12
periodic = False
frag = 122 # Bipartition fragment size

# Initialize quantum state and the bipartition
if periodic: psi = np.load('data/groundState' + str(N) + 'p.npy')
else: psi = np.load('data/groundState' + str(N) + '.npy')
bpart = np.array([int(n/frag) % 2 == 0 for n in range(N)])

# Measure partition sizes
A = 0
for spot in bpart:
    if spot: A += 1
B = N - A
print('Chain divided: (%d) -> (%d, %d)' % (N, A, B))
print(''.join(['A' if spot else 'B' for spot in bpart]))

# Set partition Hilbert dimensions
dimA = 2**A
dimB = 2**B

# Build work matrix
wmatrix = np.empty((dimA, dimB))
for i in range(dimA):
    for j in range(dimB):
        a = bits(i, A)
        b = bits(j, B)
        k = 0

        for part in bpart:
            k = k << 1
            if part: k += a.__next__()
            else: k += b.__next__()

        wmatrix[i, j] = psi[k]

u, d, v = svd(wmatrix, False)
print(-2 * np.sum(d**2 * np.log(d)))
