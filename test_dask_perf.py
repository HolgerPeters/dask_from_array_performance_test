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

from bench import softmax_with_openmp

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
def numpy_(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


@wall_times
def cython_openmp(arr):
    normalized = softmax_with_openmp(arr)
    return normalized


@wall_times
def dask_implementation(x):
    # x = da.from_array(x, chunks=x.shape[0] / CPU_COUNT, name='x')
    e_x = da.exp(x - x.max())
    out = e_x / e_x.sum()
    normalized = out.compute()
    return normalized


@wall_times
def numexpr_implementation(x):
    mx = ne.evaluate('max(x)')
    e_x = ne.evaluate('exp(x - mx)')
    sum_of_exp = ne.evaluate('sum(e_x)')
    normalized = ne.evaluate('e_x / sum_of_exp')
    return normalized


class TestPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.perfs = OrderedDict()
        N = 1e8
        cls.x = np.random.poisson(10, size=int(N)) * 1.
        cls.dx = da.from_array(cls.x, chunks=cls.x.shape[0] / CPU_COUNT, name='x')

    def testNumpy(self):
        numpy_(self.x, self.perfs, "numpy")

    def test_dask(self):
        dask_implementation(self.dx, self.perfs, "Dask")

    def testNumexpr(self):
        numexpr_implementation(self.x, self.perfs, "Numexpr")

    def testOpenMP(self):
        cython_openmp(self.x, self.perfs, "CythonOMP")

    @classmethod
    def tearDownClass(cls):
        timings = np.c_[list(cls.perfs.values())]\
                    .reshape((len(cls.perfs), -1))

        plt.boxplot(timings.T, labels=list(cls.perfs.keys()), showmeans=True)
        mx = np.percentile(timings, 97, axis=1)
        print(mx)
        assert len(mx) == len(list(cls.perfs.keys()))
        plt.ylim((0, np.max(mx)))

        plt.title("Comparison of Execution Times")
        plt.ylabel("Execution Time t in ms")
        plt.savefig("comparison.png", dpi=600)
        plt.savefig("comparison.pdf")
