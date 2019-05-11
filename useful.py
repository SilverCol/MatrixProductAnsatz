# Bit iteration
def bits(n, N):
    """ Yields the N bit sequence of an integer. """
    spot = 1 << (N-1)
    for i in range(N):
        yield (n & spot) >> (N-1)
        n = n << 1
