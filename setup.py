from distutils.core import setup
from Cython.Build import cythonize
setup(ext_modules=cythonize('chess_classic\chess_util.pyx'))