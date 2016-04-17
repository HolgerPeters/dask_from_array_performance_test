"""
Some Cython code for benchmarking
"""

cimport cython
from cython.parallel import prange
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
def csum_parallelized(np.ndarray[np.float64_t, ndim=1] arr):
    cdef int n = arr.shape[0]
    cdef int i
    cdef np.float64_t s = 0.0
    cdef double mx = np.max(arr)

    for i in prange(n, nogil=True):
        s += arr[i] / mx

    return s * mx
