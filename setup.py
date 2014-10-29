from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

include_gsl_dir = "/usr/local/include/"
lib_gsl_dir = "/usr/local/lib/"
  
     
ext = Extension("samplers", ["samplers.pyx"],
    include_dirs=[numpy.get_include(), 
                  include_gsl_dir],
    library_dirs=[lib_gsl_dir],
    libraries=["gsl","gslcblas","m"]
)
 
setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})