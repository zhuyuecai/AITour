from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [
    Extension("chess_util", ["chess_classic/chess_util.pyx"]),
    Extension("algorithms", ["algorithms/hill_climbing.pyx"],include_dirs=[numpy.get_include()],language='c++')
    
    ]
setup(name = "API for 8 queens problem",
     cmdclass = {'build_ext': build_ext},
      ext_modules=ext_modules)
