import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from useful import gstate

MAX_N = 21
for n in range(2, MAX_N):
    print('Making ground state for N = %d ...' % n)
    gstate(n, False)
    print('Repeating for periodic BC...')
    gstate(n, True)
