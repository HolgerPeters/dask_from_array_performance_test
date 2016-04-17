from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(
        "bench",
        ["bench.pyx"],
        include_dirs=[numpy.get_include()]
        # include_dirs = [...],
        # libraries = [...],
        # library_dirs = [...]
    ),
]
setup(name="daskperformancetestcase",
      ext_modules=cythonize(extensions), )
