"""
Some Cython code for benchmarking
"""

cimport cython
from cython.parallel import prange
cimport numpy as np
import numpy as np

from libc.math cimport exp

@cython.nonecheck(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def softmax_with_openmp(np.ndarray[np.float64_t, ndim=1] x):
    cdef:
        int n = x.shape[0]
        int i
        np.float64_t s = 0.0
        double max_x = np.max(x)
        np.ndarray[np.float64_t, ndim=1] e_x = np.empty(n)

    with nogil:
        for i in prange(n):
            e_x[i] = exp(x[i] - max_x)
            s += e_x[i]
        for i in prange(n):
            e_x[i] /= s
    return e_x
