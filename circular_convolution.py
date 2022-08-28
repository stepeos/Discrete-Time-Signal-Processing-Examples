import numpy as np
# circular shift operation
# [1 2 3 4] = [4 1 2 3]
def shifter(matrix):
    return np.roll(np.array(matrix),1).tolist()
#  finding circular convolution
def findCircularConvolution(x, h, N = None):
    """
    x first 1d array
    h second 1d array
    N-th circular convulution
    """
    if not N:
        N = max(len(x), len(h))
    if isinstance(x, list):
        x = x.copy()
        x += [0 for _ in range(N-len(x))]
        x = np.array(x, dtype=float)
    else:
        if x.shape[0] < N:
            x = np.concatenate(x, np.zeros((N-x.shape[0]), dtype=float))
    if isinstance(h, list):
        h = h.copy()
        h += [0 for _ in range(N-len(h))]
        h.reverse()
        h = np.array(h, dtype=float)
    else:
        if h.shape[0] < N:
            h = np.concatenate(h, np.zeros((N-x.shape[0]), dtype=float))[::-1]
    resultant = np.zeros((N), dtype=float)
    for i in range(N):
        h = np.roll(h, 1)
        resultant[i] = x @ h
    return resultant

def findLinearConvolution(x, h):
    return np.convolve(np.array(x, dtype=float), np.array(h, dtype=float)).tolist()
    
# x = [int(x) for x in input('Enter the x(n) -> ').split()]
# h = [int(x) for x in input('Enter the h(n) -> ').split()]
# x = [2, 1, 2, 1]
# h = [1, 2, 3, 4]
x = [1, 2, 3, 4, 5, 6]
h = [2, 2, 2, 1, 1, 1]
N = 8
# N = (len(x) + len(h) - 1)
circular_convolution_result = findCircularConvolution(x,h, N)
linear_convolution_result = findLinearConvolution(x,h)
print("circular/cyclic convolution: ", circular_convolution_result)
if N != (len(x) + len(h) - 1):
    print("results will differ")
else:
    print("results shouldn't differ")
print("linear/cyclic convolution: ", linear_convolution_result)
