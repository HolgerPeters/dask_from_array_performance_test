import time
import unittest
from collections import OrderedDict, namedtuple
from multiprocessing import cpu_count

import dask.array as da
import numexpr as ne
import numpy as np
from matplotlib import use
use('agg')
from matplotlib import pyplot as plt

from bench import csum_par

CPU_COUNT = cpu_count()


def wall_times(method):
    """Decorator that executes a function multiple times, recording
    wall-times"""

    def timed(x, perfs, label):
        timings = []
        for i in range(10):
            tik = time.time()
            result = method(x)
            tok = time.time()

            wall_time = 1000 * (tok - tik)
            timings.append(wall_time)
        perfs[label] = np.asarray(timings)
        return result

    return timed


@wall_times
def numpy_(arr):
    mx = arr.max()
    (arr / mx).sum()


@wall_times
def cython_naive():
    csum(arr)


@wall_times
def cython_openmp(arr):
    csum_par(arr)


@wall_times
def dask_(arr):
    darr = da.from_array(arr, chunks=arr.shape[0] / CPU_COUNT)
    mx = darr.max()
    x = (darr / mx).sum() * mx
    x.compute()


@wall_times
def dask_native(darr):
    mx = darr.max()
    x = (darr / mx).sum() * mx
    x.compute()


@wall_times
def dask_no_computation(arr):
    darr = da.from_array(arr, chunks=arr.shape[0] / CPU_COUNT)


@wall_times
def numexpr_(arr):
    mx = ne.evaluate('max(arr)')
    ne.evaluate('sum(arr / mx)') * mx


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perfs = OrderedDict()
        N = 5e8
        cls.x = np.random.poisson(10, size=int(N)) * 1.
        cls.dx = da.from_array(cls.x, chunks=cls.x.shape[0] / CPU_COUNT)

    def testNumpy(self):
        numpy_(self.x, self.perfs, "np")

    def test_dask(self):
        dask_(self.x, self.perfs, "DskFromNumpy")

    def testdask_from_array_only(self):
        dask_no_computation(self.x, self.perfs, "DskFromArray")

    def test_dask_native(self):
        dask_native(self.dx, self.perfs, "DskNative")

    def testNumexpr(self):
        numexpr_(self.x, self.perfs, "NX")

    def testOpenMP(self):
        cython_openmp(self.x, self.perfs, "CyOMP")

    @classmethod
    def tearDownClass(cls):
        timings = np.c_[list(cls.perfs.values())]\
                    .reshape((len(cls.perfs), -1))

        plt.boxplot(timings.T, labels=list(cls.perfs.keys()), showmeans=True)
        plt.ylim((0, timings.max()))

        plt.title("Comparison of Execution Times")
        plt.ylabel("Execution Time t in ms")
        plt.savefig("comparison.png", dpi=600)
        plt.savefig("comparison.pdf")
