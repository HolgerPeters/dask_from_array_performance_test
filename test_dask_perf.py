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

from bench import softmax_openmp

CPU_COUNT = cpu_count()


def wall_times(method):
    """Decorator that executes a function multiple times, recording
    wall-times. This decorator is meant to give a very rough idea about
    performance."""

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

def softmax_numpy(x):
    e_x = np.exp(x - x.max())
    return e_x / e_x.sum()


def softmax_dask(x):
    # x = da.from_array(x, chunks=x.shape[0] / CPU_COUNT, name='x')
    e_x = da.exp(x - x.max())
    out = e_x / e_x.sum()
    normalized = out.compute()
    return normalized


def softmax_numexpr(x):
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

    def test_numpy(self):
        wall_times(softmax_numpy)(self.x, self.perfs, "numpy")

    def test_dask(self):
        wall_times(softmax_dask)(self.dx, self.perfs, "Dask")

    def test_numexpr(self):
        wall_times(softmax_numexpr)(self.x, self.perfs, "Numexpr")

    def test_openmp(self):
        wall_times(softmax_openmp)(self.x, self.perfs, "CythonOMP")

    @classmethod
    def tearDownClass(cls):
        timings = np.c_[list(cls.perfs.values())]\
                    .reshape((len(cls.perfs), -1))

        plt.boxplot(timings.T, labels=list(cls.perfs.keys()), showmeans=True)
        mx = np.percentile(timings, 97, axis=1)
        print(mx)
        plt.ylim((0, np.max(mx)))

        plt.title("Comparison of Execution Times")
        plt.ylabel("Execution Time t in ms")
        plt.savefig("comparison.png", dpi=600)
        plt.savefig("comparison.pdf")


class SoftmaxTestSuite(object):
    """Contains generic tests for all softmax implementations"""


    def test_softmax_of_array_of_ones_equalsmean(self):
        x = np.ones(5)
        result = self.softmax(x)
        np.testing.assert_allclose(result, 1./5)

    def test_short_vector(self):
        x = np.r_[np.log(1), np.log(2), np.log(3), np.log(4)]
        result = self.softmax(x)

        s = sum(range(5))
        np.testing.assert_allclose(np.r_[1., 2., 3., 4.] /  s, result)


class TestNumpy(unittest.TestCase, SoftmaxTestSuite):
    def setUp(self):
        self.softmax = softmax_numpy


class TestNumexpr(unittest.TestCase, SoftmaxTestSuite):
    def setUp(self):
        self.softmax = softmax_numexpr


class TestOpenMP(unittest.TestCase, SoftmaxTestSuite):
    def setUp(self):
        self.softmax = softmax_openmp


class TestDask(unittest.TestCase, SoftmaxTestSuite):
    def setUp(self):
        self.softmax = softmax_dask
