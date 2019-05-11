import numpy as np

# Parameters
N = 12
periodic = False
frag = 1 # Bipartition fragment size

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

# Build work matrix
# TODO
