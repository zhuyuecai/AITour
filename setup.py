from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("chess_util", ["chess_classic/chess_util.pyx"])

    
    ]
setup(name = "API for 8 queens problem",
     cmdclass = {'build_ext': build_ext},
      ext_modules=ext_modules)
