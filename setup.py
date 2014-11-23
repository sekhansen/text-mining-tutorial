from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy
  
     
ext = Extension("samplers", ["samplers.pyx"],
    include_dirs=[numpy.get_include(), 
                  cython_gsl.get_cython_include_dir()],
    library_dirs=[cython_gsl.get_library_dir()],
    libraries=libraries=cython_gsl.get_libraries()
)
 
setup(name = "samplers",
	ext_modules=[ext],
    cmdclass = {'build_ext': build_ext}
	include_dirs = [cython_gsl.get_include())
