import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

# Problem size
N = 4
DIM = 2**N
periodic = False 
# Hamiltonian matrix
H = lil_matrix((DIM, DIM))

# Bit iteration
def bits(n):
    for i in range(N):
        yield n & 1
        n = n >> 1

# Hamiltonian construction
for row in range(DIM):
    for n, bit in enumerate(bits(row)): 
        if n == 0:
            if periodic: first = bit
            b1 = bit
        else:
            if b1 == bit:
                H[row, row] += 1
            else:
                H[row, row] -= 1
                col = row ^ (3 << (n-1))
                H[row, col] += 2
            b1 = bit
    if periodic:
        if b1 == first:
            H[row, row] += 1
        else:
            H[row, row] -= 1
            col = row ^ (1 << (N-2) + 1)
            H[row, col] += 2

print(H.toarray())

w, v = eigsh(H, k=1, which='SA')
print(w)
print(v)
np.save('data/groundState' + str(N) + '.npy', v)
