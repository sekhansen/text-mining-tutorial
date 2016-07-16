from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
# import sys

# See GSL Library
# if sys.platform == "win32":
#     include_gsl_dir = sys.exec_prefix.lower().split("anaconda")[0] + \
#         "anaconda\\gsl\\include"
#     lib_gsl_dir = sys.exec_prefix.lower().split("anaconda")[0] + \
#         "anaconda\\gsl\\lib"
# else:
#     include_gsl_dir = sys.exec_prefix+"\\include"
#     lib_gsl_dir = sys.exec_prefix+"\\lib"

include_gsl_dir = "/usr/local/include/"
lib_gsl_dir = "/usr/local/lib/"

# <<<<<<< HEAD
# =======
# ext = Extension("topicmodels.samplers.samplers_lda",
#                 ["topicmodels/samplers/samplers_lda.pyx"],
#                 include_dirs=[numpy.get_include(), include_gsl_dir],
#                 library_dirs=[lib_gsl_dir],
#                 libraries=["gsl", "gslcblas", "m"])

# >>>>>>> origin/master

##########
# comment code block below if not using GSL
##########
ext = Extension("topicmodels.samplers.samplers_lda",
                ["topicmodels/samplers/samplers_lda.pyx"],
                include_dirs=[numpy.get_include(), include_gsl_dir],
                library_dirs=[lib_gsl_dir],
                libraries=["gsl", "gslcblas", "m"])

########
# uncomment code block below to use numpy only without GSL
########

# ext = Extension("topicmodels.samplers.samplers_lda",
#                 ["topicmodels/samplers/samplers_lda.pyx"],
#                 include_dirs=[numpy.get_include()])

setup(name="topicmodels",
      ext_modules=[ext],
      packages=['topicmodels', 'topicmodels.LDA', 'topicmodels.samplers'],
      package_data={'topicmodels': ['*.txt']},
      cmdclass={'build_ext': build_ext})
