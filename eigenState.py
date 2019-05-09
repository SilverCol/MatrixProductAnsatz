import numpy as np

N = 4
DIM = 2**N

def bits(n):
    for i in range(N):
        yield n & 1
        n = n >> 1

for row in range(DIM):
    for n, bit in enumerate(bits(row)): 
        if n == 0:
            first = bit
            b1 = bit
        else:


