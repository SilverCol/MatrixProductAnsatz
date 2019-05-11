# Bit iteration
def bits(n, N):
    """ Yields the N bit sequence of an integer. """
    for i in range(N):
        yield n & 1
        n = n >> 1
