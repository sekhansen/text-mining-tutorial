from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy  
     
#Use cython_gsl to interact with GSL 
import cython_gsl

ext = Extension("samplers", ["samplers.pyx"],
	libraries = cython_gsl.get_libraries(),
    include_dirs=[numpy.get_include(), 
                  cython_gsl.get_include()],
    library_dirs=[cython_gsl.get_cython_include_dir()])
 
setup(ext_modules=[ext],
	include_dirs = [cython_gsl.get_include()],
    cmdclass = {'build_ext': build_ext})